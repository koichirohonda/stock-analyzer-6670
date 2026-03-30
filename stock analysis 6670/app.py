"""Flask web application for the SEC Filing Analyzer."""

import os

from flask import Flask, Response, render_template, request

from agents import run_analysis

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze")
def analyze():
    company = request.args.get("company", "").strip()
    if not company:
        return {"error": "Missing 'company' parameter"}, 400

    return Response(
        run_analysis(company),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable before starting.")
        raise SystemExit(1)

    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
