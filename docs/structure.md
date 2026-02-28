# Repository Structure Guide

## Purpose

Keep the repository navigable while preserving data and model assets required by this project.

## Root Directory Policy

- Keep only entry files in root:
- `README.md`
- `AGENTS.md`
- active blueprint docs (`项目架构蓝图_*`, `业务流程图_*`, `升级*.md`)
- Do not place temporary reports or run logs in root.

## Directory Roles

- `backend/`: backend source code and backend-local config.
- `frontend_new/`: frontend source code.
- `scripts/`: runnable utility scripts.
- `docs/`: long-term documentation.
- `docs/archive/YYYY-MM/`: historical reports and obsolete docs.
- `logs/`: runtime and evaluation logs.
- `logs/archive/YYYY-MM/`: archived root logs and older run outputs.
- `data/`: datasets and training data (kept in repo by project decision).
- `models/`: local models/weights (kept in repo by project decision).
- external repos (`GPTQModel`, `llama.cpp`, `local_llama_factory`):
  moved to `/home/kkk/Project/`, see `docs/external_repositories.md`.

## Rules For New Files

- Reports with timestamp must go to `docs/archive/YYYY-MM/`.
- RCA/debug artifacts must go to `logs/rca/<topic>/`.
- One-off local outputs should go to `logs/archive/YYYY-MM/` instead of root.
- If a document becomes actively maintained, move it from archive back to `docs/`.

## No-Touch Code Scope During Cleanup

The cleanup process should not modify core runtime code:

- `backend/app/`
- `backend/main.py`
- `frontend_new/app/`
- `frontend_new/components/`
