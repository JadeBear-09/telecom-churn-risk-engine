# Commit Guide

This repo uses Conventional Commits so history stays searchable and release notes are
easy to generate.

## Format

```text
type(scope): short summary
```

Use imperative mood and keep the subject near 50 characters.

## Common Types

| Type | Use when |
| --- | --- |
| `feat` | adding user-visible behavior |
| `fix` | correcting broken behavior |
| `docs` | changing README, reports, or developer docs |
| `test` | adding or updating tests |
| `refactor` | restructuring code without behavior change |
| `perf` | improving runtime, latency, or memory |
| `build` | changing Docker, packaging, or dependency setup |
| `ci` | changing GitHub Actions or automation |
| `chore` | maintenance without product behavior change |

## Suggested Scopes

- `api`
- `dashboard`
- `training`
- `features`
- `thresholds`
- `monitoring`
- `docs`
- `ci`

## Examples

```text
feat(api): add batch risk scoring endpoint
fix(thresholds): apply segment fallback threshold
docs(readme): add model snapshot and runbook
test(api): cover invalid batch upload rejection
ci: train artifacts before pytest
```

## PR Commit Hygiene

- Prefer a small set of meaningful commits over many save-point commits.
- Squash local experiments before review.
- Mention metric impact in the commit body when model behavior changes.
- Never commit `.env`, API keys, raw private customer data, or generated secrets.
