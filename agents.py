"""LangGraph multi-agent pipeline for SEC filing analysis."""

import json
import operator
from typing import Annotated, TypedDict

import anthropic
from langgraph.graph import END, StateGraph

from sec_edgar import download_filing, fetch_filing_index, get_cik

# ── Config ───────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
REQUIRED_METRICS = [
    "revenue",
    "net_income",
    "eps",
    "operating_income",
    "total_assets",
    "total_liabilities",
    "equity",
]

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


def call_claude(system_prompt: str, user_content: str, max_tokens: int = 2048) -> str:
    """Single entry-point for all Claude API calls."""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text


# ── Shared state ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    company: str
    question: str
    filing_index: list[dict]
    filing_texts: list[dict]  # [{filename, text}]
    metrics_data: list[dict]
    research: str
    analysis: str
    final_answer: str
    missing_metrics: list[str]
    retry_count: int
    messages: Annotated[list, operator.add]


# ── Agents ───────────────────────────────────────────────────────────────────
def fetcher(state: AgentState) -> dict:
    """Resolve company CIK, fetch filing index, download 10-Q texts."""
    question = state["question"]

    # Extract company name via Claude
    company = call_claude(
        "Extract only the company name from the question. Reply with just the company name, nothing else.",
        question,
    ).strip()

    cik, title = get_cik(company)
    if not cik:
        msg = f"No SEC filings found for '{company}'."
        return {
            "company": company,
            "filing_index": [],
            "filing_texts": [],
            "messages": [("fetcher", msg)],
        }

    index = fetch_filing_index(cik)

    # Download 10-Q filings (most useful for quarterly metrics)
    filing_texts = []
    for entry in index:
        if entry["form"] != "10-Q":
            continue
        text = download_filing(entry["url"])
        if text:
            filing_texts.append({"filename": f"{entry['date']}_{entry['form']}", "text": text})

    summary = (
        f"Found {len(index)} filing(s) for {title} (CIK {cik.lstrip('0')}). "
        f"Downloaded {len(filing_texts)} 10-Q document(s) for analysis."
    )
    return {
        "company": company,
        "filing_index": index,
        "filing_texts": filing_texts,
        "messages": [("fetcher", summary)],
    }


def researcher(state: AgentState) -> dict:
    """Extract financial metrics from each filing via chunked LLM analysis."""
    missing = state.get("missing_metrics", [])
    focus = (
        f"The analyst flagged these metrics as missing: {', '.join(missing)}. "
        "Search specifically for these values.\n"
        if missing
        else ""
    )

    # Preserve existing metrics across retries
    metrics_by_file: dict[str, dict] = {
        entry["source_file"]: entry for entry in state.get("metrics_data", [])
    }

    files = state.get("filing_texts", [])
    if not files:
        fallback = "No filing texts available for analysis."
        return {
            "metrics_data": [],
            "research": fallback,
            "messages": [("researcher", fallback)],
        }

    system_prompt = (
        "You are a quantitative financial research assistant analysing a 10-Q filing chunk. "
        "Analyse ONLY the data explicitly present in the text below. "
        "Do NOT use any prior knowledge or external information. "
        "Extract any of these metrics found in this chunk: "
        "revenue, net_income, eps, operating_income, operating_cash_flow, free_cash_flow, "
        "total_assets, total_liabilities, equity, gross_margin. "
        f"{focus}"
        "Reply with a single valid JSON object and NOTHING else. Use null for any "
        "metric not found in this chunk. Include a 'period' key if the reporting period is mentioned.\n"
        'Example: {"period":null, "revenue":null, "net_income":null, "eps":null, '
        '"operating_income":null, "total_assets":null, "total_liabilities":null, '
        '"equity":null, "gross_margin":null}'
    )

    CHUNK_SIZE = 12_000
    metric_keys = [
        "revenue", "net_income", "eps", "operating_income",
        "operating_cash_flow", "free_cash_flow", "total_assets",
        "total_liabilities", "equity", "gross_margin",
    ]

    for filing in files:
        filename = filing["filename"]
        text = filing["text"]
        chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

        merged: dict = {"source_file": filename}

        for idx, chunk in enumerate(chunks[:5]):
            raw = call_claude(system_prompt, f"Filing chunk:\n{chunk}")

            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            try:
                chunk_metrics = json.loads(raw)
            except json.JSONDecodeError:
                continue

            for key, val in chunk_metrics.items():
                if val is not None and merged.get(key) is None:
                    merged[key] = val

            if all(merged.get(k) is not None for k in metric_keys):
                break

        metrics_by_file[filename] = merged

    metrics_data = list(metrics_by_file.values())
    research_summary = json.dumps(metrics_data, indent=2)
    return {
        "metrics_data": metrics_data,
        "research": research_summary,
        "messages": [("researcher", research_summary)],
    }


def analyst(state: AgentState) -> dict:
    """Check completeness of extracted metrics and provide analysis."""
    required = ", ".join(REQUIRED_METRICS)
    text = call_claude(
        (
            "You are a senior financial analyst. Given research notes, do two things:\n"
            "1. Check which of these required metrics are present with actual numbers: "
            f"{required}.\n"
            "2. Provide a structured analysis of the metrics that ARE present.\n\n"
            "At the very end of your response, on its own line, write exactly:\n"
            "MISSING_METRICS: <comma-separated list of missing metrics, or 'none'>"
        ),
        f"Question: {state['question']}\n\nResearch notes:\n{state['research']}",
    )

    missing: list[str] = []
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.upper().startswith("MISSING_METRICS:"):
            raw = line.split(":", 1)[1].strip()
            if raw.lower() != "none":
                missing = [m.strip() for m in raw.split(",") if m.strip()]
            break

    retry_count = state.get("retry_count", 0)
    return {
        "analysis": text,
        "missing_metrics": missing,
        "retry_count": retry_count + 1,
        "messages": [("analyst", text)],
    }


def summarizer(state: AgentState) -> dict:
    """Produce a concise final answer from the analysis."""
    text = call_claude(
        (
            "You are a financial writer. Summarize the analysis into a clear, "
            "concise answer (3-5 sentences) suitable for a non-expert audience."
        ),
        f"Question: {state['question']}\n\nAnalysis:\n{state['analysis']}",
    )
    return {
        "final_answer": text,
        "messages": [("summarizer", text)],
    }


# ── Router ───────────────────────────────────────────────────────────────────
def route_analyst(state: AgentState) -> str:
    if state.get("missing_metrics") and state.get("retry_count", 0) < MAX_RETRIES:
        return "researcher"
    return "summarizer"


# ── Graph ────────────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("fetcher", fetcher)
    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("summarizer", summarizer)

    graph.set_entry_point("fetcher")
    graph.add_edge("fetcher", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_conditional_edges(
        "analyst",
        route_analyst,
        {"researcher": "researcher", "summarizer": "summarizer"},
    )
    graph.add_edge("summarizer", END)

    return graph.compile()


# ── SSE streaming helper ────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def run_analysis(company: str):
    """Generator that yields SSE-formatted events as each agent completes."""
    graph = build_graph()

    initial_state = {
        "company": company,
        "question": f"Analyse the latest SEC filings for {company}.",
        "filing_index": [],
        "filing_texts": [],
        "metrics_data": [],
        "research": "",
        "analysis": "",
        "final_answer": "",
        "missing_metrics": [],
        "retry_count": 0,
        "messages": [],
    }

    yield _sse({"agent": "fetcher", "status": "running"})

    try:
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            node_name, update = next(iter(chunk.items()))
            if node_name == "fetcher":
                yield _sse({
                    "agent": "fetcher",
                    "status": "complete",
                    "data": {
                        "company": update.get("company", company),
                        "filings": update.get("filing_index", []),
                        "downloaded": len(update.get("filing_texts", [])),
                    },
                })
                yield _sse({"agent": "researcher", "status": "running"})

            elif node_name == "researcher":
                yield _sse({
                    "agent": "researcher",
                    "status": "complete",
                    "data": {"metrics": update.get("metrics_data", [])},
                })
                yield _sse({"agent": "analyst", "status": "running"})

            elif node_name == "analyst":
                missing = update.get("missing_metrics", [])
                retry = update.get("retry_count", 0)
                yield _sse({
                    "agent": "analyst",
                    "status": "complete",
                    "data": {
                        "analysis": update.get("analysis", ""),
                        "missing": missing,
                        "retry": retry,
                    },
                })
                if missing and retry < MAX_RETRIES:
                    yield _sse({"agent": "researcher", "status": "retry", "data": {"missing": missing}})
                else:
                    yield _sse({"agent": "summarizer", "status": "running"})

            elif node_name == "summarizer":
                yield _sse({
                    "agent": "summarizer",
                    "status": "complete",
                    "data": {"summary": update.get("final_answer", "")},
                })

    except Exception as e:
        yield _sse({"agent": "error", "status": "error", "data": {"message": str(e)}})

    yield _sse({"agent": "done"})
