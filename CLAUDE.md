# polarq — Claude Code instructions

## Python environment

Always use the project-local virtual environment at `.venv/`:

```bash
.venv/bin/python   # interpreter
.venv/bin/pip      # package installer
.venv/bin/pytest   # test runner
```

If it is missing, set up new onw with `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

Never use the system `python3` or `pip3` for this project.

## Running tests

```bash
.venv/bin/python -m pytest tests/ -v
```
