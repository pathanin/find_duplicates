# Daily Improvement Log

One row per run. Older rows are moved to `task-archive.md` once this table exceeds 30 rows — see bottom of file.

| Date | Outcome | Summary | Branch @ Commit | Tests |
|---|---|---|---|---|
| 2026-07-15 | Done | CLAUDE.md test list omitted test_score_group.py, causing automated runs to silently skip it; added it | docs/add-missing-test-to-claude-md @ 11ece6a | 7/7 pass |
| 2026-07-15 | Done | analyze_paths's ANALYZE_PARALLEL_THRESHOLD boundary and precomputed_stats param had zero test coverage since 9b6a6bd; added 3 tests | test/analyze-parallel-threshold-coverage @ bac75b1 | 8/8 pass |
| 2026-07-15 | Done | _unapply used rename() (broke cross-device --dest) and dropped its manifest entry on partial restore failure; fixed both, matching _apply's safety | fix/unapply-cross-device-and-partial-restore @ 38f8b53 | 9/9 pass |
| 2026-07-16 | Done | CLAUDE.md test list omitted test_unapply_crash_safety.py (added 38f8b53), same recurring omission class as 07-15; added it | docs/add-missing-unapply-test-to-claude-md @ 765d75f | 8/8 pass |
