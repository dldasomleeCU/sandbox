# Sandbox Support EDA

Lightweight, interview-ready exploration of a customer support tickets dataset.

What it does
- Cleans and validates timestamps; filters negative time-to-resolution (TTR)
- Summarizes ticket volume by type; identifies ~80% head categories
- Compares channel mix overall vs within head types
- Extracts recurring themes for a chosen type × channel using OpenAI with batching + merge
- Saves artifacts: `snippets.csv`, `batch_summaries.csv`, `final_themes.csv`

Quick start
- Python 3.10+
- Install deps: `pip install -r requirements.txt` (or see minimal list below)
- Set `OPENAI_API_KEY` in your environment

Run
- Default run: `python sandbox.py`
- Faster run: `python sandbox.py --fast`
- Deterministic merge: `python sandbox.py --temp_merge 0.0`
- Local CSVs: `python sandbox.py --path /path/to/csvs`

Minimal dependencies
- pandas, numpy, python-dotenv, openai
- Optional (plots): seaborn, matplotlib
- Optional (download): kagglehub

Notes
- The LLM prompts ask for estimated prevalence per theme. For speed, use `--fast` or reduce `--N` and increase `--batch_size`.

