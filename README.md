# round-table-portfolio

Multi-agent debate panel for medium-term US-equity portfolio construction, with per-agent counterfactual tracking and quarterly meta-review.

> Built with [Claude Code](https://claude.ai/code)

---

## What it does

Seven distinct analyst personas (Value, Growth, Discretionary Macro, CTA-Systematic Macro, Technical, Quant-Systematic, Risk Officer) each research the full S&P 500 universe independently each week using live web search and on-demand data tools, then debate a shortlisted set of names across two rounds. The panel produces one consensus portfolio recommendation plus seven per-agent counterfactual portfolios. A local SQLite ledger tracks every holding, weekly return, debate transcript, and agent stance from Week 1 onward — giving you a durable track record that grows more valuable over time.

The founder reviews and approves (or overrides) the panel's proposed delta each week inside a Claude Code session. A separate read-only Streamlit dashboard displays performance, debate history, and the quarterly meta-review.

## Project status

PoC — M1 in progress.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
python -m round_table_portfolio.storage.apply_schema
```

## Running the tests

```bash
source .venv/bin/activate
SKIP_LIVE=1 python -m pytest -v
```

## Environment variables

See `.env.example` for the full list with descriptions.

## License

MIT
