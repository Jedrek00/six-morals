.PHONY: install format run-demo

install:
	uv sync

format:
	uv run ruff format

run-demo:
	uv run python -m streamlit run app.py