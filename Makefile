# Makefile — AI Suspect Sketch Generator

.PHONY: install run-ui run-api test lint clean

install:
	pip install -r requirements.txt

run-ui:
	streamlit run ui/app.py

run-api:
	uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

test-nlp:
	pytest tests/test_nlp_parser.py -v

test-prompt:
	pytest tests/test_prompt_engineer.py -v

# Quick end-to-end smoke test (no API keys needed — uses rule-based)
smoke:
	python -c "\
from nlp.nlp_parser import extract_attributes_rule_based; \
from pipeline.prompt_engineer import build_forensic_prompt; \
attrs = extract_attributes_rule_based('White male, 40s, square jaw, scar on cheek'); \
p, n = build_forensic_prompt(attrs); \
print('PROMPT:', p[:120]); \
print('NEGATIVE:', n[:80]); \
print('OK')"

lint:
	python -m py_compile nlp/nlp_parser.py pipeline/prompt_engineer.py \
	    pipeline/generation_pipeline.py api/api.py api/models.py ui/app.py
	@echo "Syntax OK"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "Cleaned"
