# Video Streaming — Performance Bugs (April 2026)

This bug has surfaced twice. Symptom: video takes a long time to start playing and buffers repeatedly mid-playback.

## Root Causes and Fixes

All fixes are in `api/routes.py`.

| # | Scope | Root cause | Fix |
|---|-------|-----------|-----|
| 1 | All videos | `_get_video_path()` called synchronously from async route — on cache miss it does a full scan of all ChromaDB documents (`get_vector_store_data()` iterates every metadata record), blocking the uvicorn event loop | Added `_get_video_path_async()` which runs the scan in a thread via `run_in_executor` |
| 2 | WebDAV videos | HEAD request to OwnCloud on **every** Range request (browser sends 10–30 per video) | Added `_video_size_cache: dict[str, int]` — HEAD is done once per video ID, result reused |
| 3 | WebDAV videos | `httpx.AsyncClient(timeout=60)` cuts off large videos mid-stream | Changed to `httpx.Timeout(connect=10.0, read=None)` — no read timeout |
| 4 | WebDAV videos | No `Content-Length` on initial (non-Range) response — browser can't show seek bar or buffer progressively | `Content-Length` now always included when file size is known |

## Key Invariants to Preserve

- `_video_cache` (filepath by video_id) and `_video_size_cache` (file size by video_id) are module-level in-process dicts. They are **not** invalidated on video deletion — a server restart clears them. That is acceptable.
- `_get_video_path()` must remain a plain `def` (not async) because it is also called from synchronous contexts. The async wrapper `_get_video_path_async()` is what the route uses.
- The WebDAV branch is entered when `not os.path.isabs(video_path)` — a relative path means the file lives on OwnCloud, not local disk. Local-disk files are served directly via `aiofiles` with proper range support; they do not go through `httpx`.
