# Contributing

## Development Setup

Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Project Rules

- Preserve chronological data cutoffs. Pregame features must never include the
  target game's outcome.
- Add tests for schema, prediction, or parameter changes.
- Keep generated data, cached API responses, models, and reports out of Git.
- Do not commit credentials, private endpoints, or machine-specific paths.
- Document any change that alters model behavior or evaluation methodology.
- Compare new models against the frozen V1 and selected V2 baselines on future
  dates, not random train/test splits.

## Pull Requests

Describe the motivation, behavioral change, validation period, metrics, and
tests run. A lower aggregate error is not sufficient if bias, tail errors,
coverage, or leakage controls regress.

## Data Issues

When reporting an endpoint/schema issue, include the endpoint class, season,
game ID when applicable, expected columns, actual columns, and whether the
response came from cache. Do not attach copyrighted bulk data or credentials.
