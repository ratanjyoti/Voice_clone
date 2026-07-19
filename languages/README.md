# Language Workspaces

Use these folders when you want to work on one language without searching the whole repository.

The files are indexed here, not moved. That keeps existing generation/evaluation scripts working.

## Folders
- `english/`: English models, data, outputs, results, and docs.
- `hindi/`: Hindi models, data, outputs, results, and docs.
- `arabic/`: Arabic models, data, outputs, results, and docs.
- `shared/`: cross-language scripts, configs, reports, dashboard files, and third-party repository pointers.

## Counts
| Workspace | Files |
| --- | --- |
| english | 226 |
| hindi | 128 |
| arabic | 603 |
| shared | 295 |

## Rebuild
```powershell
python scripts\build_language_workspaces.py
```
