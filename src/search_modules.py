# from __future__ import annotations

# from pathlib import Path
# from typing import Any, Dict, List

# from loguru import logger


# class SearchModulesHandler:
#     """Simple local search over Python modules in this repo.

#     This is intentionally lightweight: it searches file names + first lines
#     for a query and returns matching module paths.
#     """

#     name = "search_modules"
#     description = "Search local Python modules by filename/content snippet."

#     def __init__(self, root: str | None = None):
#         repo_root = Path(root) if root else Path(__file__).resolve().parents[1]
#         self._src_root = repo_root / "src"

#     def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
#         query = str((arguments or {}).get("query") or "").strip()
#         limit = (arguments or {}).get("limit", 10)

#         try:
#             limit_int = int(limit)
#         except Exception:
#             limit_int = 10

#         if not query:
#             return {"content": {"error": "Missing required field 'query'"}}

#         if not self._src_root.exists():
#             return {"content": {"matches": [], "note": "src/ folder not found"}}

#         matches: List[Dict[str, Any]] = []
#         q = query.lower()

#         for py_file in self._src_root.rglob("*.py"):
#             rel = py_file.relative_to(self._src_root)
#             rel_str = str(rel).replace("\\", "/")

#             hit = q in rel_str.lower()
#             first_lines = ""
#             if not hit:
#                 try:
#                     with py_file.open("r", encoding="utf-8") as f:
#                         first_lines = "".join([next(f) for _ in range(20)])
#                     hit = q in first_lines.lower()
#                 except StopIteration:
#                     hit = q in first_lines.lower()
#                 except Exception:
#                     # ignore unreadable files
#                     pass

#             if hit:
#                 matches.append(
#                     {
#                         "module": rel_str.removesuffix(".py").replace("/", "."),
#                         "path": f"src/{rel_str}",
#                     }
#                 )

#             if len(matches) >= max(1, min(limit_int, 50)):
#                 break

#         logger.info(f"search_modules query='{query}' matches={len(matches)}")
#         return {"content": {"query": query, "matches": matches}}
