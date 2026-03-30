"""SEC EDGAR API helpers for fetching company filings."""

import re
import datetime
import time
import requests

SEC_HEADERS = {"User-Agent": "MultiAgentResearch agent@example.com"}


def get_cik(company_name: str) -> tuple[str | None, str | None]:
    """Return (zero-padded CIK, company title) for the closest match.

    Matches against both company title and ticker symbol so users can
    type "AAPL" or "Apple".
    """
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()

    name_lower = company_name.strip().lower()
    for entry in resp.json().values():
        if (
            name_lower in entry["title"].lower()
            or name_lower == entry["ticker"].lower()
        ):
            return str(entry["cik_str"]).zfill(10), entry["title"]
    return None, None


def fetch_filing_index(cik: str) -> list[dict]:
    """Return a list of 10-K / 10-Q / 8-K filings for the current year.

    Each dict: {form, date, accession, primary_doc, url}
    """
    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=SEC_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()

    recent = resp.json()["filings"]["recent"]
    forms = recent["form"]
    dates = recent["filingDate"]
    accessions = recent["accessionNumber"]
    primary_docs = recent.get("primaryDocument", [""] * len(forms))

    current_year = str(datetime.date.today().year)
    target_forms = {"10-K", "10-Q", "8-K"}
    results = []

    for i, form in enumerate(forms):
        if form not in target_forms:
            continue
        if not dates[i].startswith(current_year):
            continue
        doc = primary_docs[i] if i < len(primary_docs) else ""
        if not doc:
            continue

        acc_nodash = accessions[i].replace("-", "")
        cik_trimmed = cik.lstrip("0")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_trimmed}/{acc_nodash}/{doc}"

        results.append({
            "form": form,
            "date": dates[i],
            "accession": accessions[i],
            "primary_doc": doc,
            "url": url,
        })

    return results


_TAG_RE = re.compile(r"<[^>]+>")


def download_filing(url: str) -> str:
    """Download a filing document and return its plain-text content."""
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text
        # Strip HTML tags for a rough plain-text conversion
        text = _TAG_RE.sub(" ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        time.sleep(0.5)  # respect SEC rate limits
        return text
    except Exception:
        return ""
