# Contributing

Thanks for improving Telecom Churn Risk Engine. Keep changes small, tested, and easy to
review.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
make install
make train
make test
```

## Development Workflow

1. Create a branch from `main`.
2. Keep one pull request focused on one behavior or documentation goal.
3. Update tests, reports, or sample payloads when behavior changes.
4. Run the validation commands before opening a PR.
5. Use the commit style in [docs/COMMIT_GUIDE.md](docs/COMMIT_GUIDE.md).

## Validation Checklist

```bash
make test
python -m src.evaluate
python -m src.batch_scoring
```

Use `make train` when changes touch features, thresholds, model code, metrics, or
generated artifacts.

## Pull Request Expectations

- Explain what changed and why.
- Include screenshots or sample API output for UI/API-facing changes.
- Call out metric movement when model behavior changes.
- Keep generated data out of the PR unless it is an intentional artifact.
- Avoid committing secrets, private customer data, or raw production records.
