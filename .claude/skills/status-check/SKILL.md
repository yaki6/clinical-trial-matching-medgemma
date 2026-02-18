---
name: status-check
description: Check project status and find what to work on next. Use at session start or when asked "what's next?"
triggers:
  - status
  - what's next
  - project status
  - where are we
---

# Project Status Check

## Quick Status

1. Read the dashboard:
   ```
   Read docs/status/DASHBOARD.md
   ```

2. Check freshness — if `<!-- Last updated: ... -->` is > 48h old:
   ```bash
   git log --oneline -10
   ```
   Then update DASHBOARD.md to match reality.

3. Check BDD progress (if BDD framework is set up):
   ```bash
   # Count implemented vs WIP scenarios
   uv run pytest tests/bdd/ --collect-only -q 2>/dev/null | tail -5

   # Count @wip scenarios
   uv run pytest tests/bdd/ -m wip --collect-only -q 2>/dev/null | tail -5
   ```

4. Check for unanswered human questions in "Open Questions for Human" section. If items are > 3 sessions old, flag to user.

## Finding What's Next

1. Look at "Current Sprint Goals" in DASHBOARD.md — pick first unchecked item
2. Check "Blockers" — skip blocked tasks
3. Prefer: unblocked tasks → lowest dependency depth → P0 components first

## Component Dependency Order

```
data/ → models/ → ingest/ → prescreen/ → validate/ → evaluation/ → cli/
```

Start from the left. Don't implement downstream components before upstream ones pass tests.

## Report Format

When reporting status to the user, include:
- Current phase
- Sprint goals completion (X of Y done)
- Any blockers or unanswered questions
- Recommended next task
