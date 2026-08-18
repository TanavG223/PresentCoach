# Contributing

Contributions are welcome. Keep the local-first privacy boundary intact and do
not commit recordings, profile archives, model weights, Keychain exports, or
real transcripts.

Before opening a pull request, run:

```bash
npm --prefix frontend install
npm --prefix frontend run build
python -m pytest
python -m pip check
```

Changes to measurement thresholds must include raw-number rationale and a
repeatability test. Changes to LLM prompts or validators must add adversarial
cases and rerun `scripts/evaluate_presentcoach_llm.py` against the local model.
