# Security Issues — Fixed

| # | Severity | Issue | Fix location |
|---|----------|-------|--------------|
| 1 | CRITICAL | Hardcoded MySQL password `M00dl3` in source | `routes.py`, `export_to_owncloud.py`, `eval/01_seed_annotations.py` → `os.getenv("MOODLE_DB_PASSWORD")` |
| 2 | CRITICAL | No authentication on any backend endpoint | `server.py` — `require_internal_token` middleware + `X-Internal-Token` header |
| 3 | CRITICAL | Missing CSRF on `chat_proxy.php` | `chat_proxy.php` — `confirm_sesskey()` on JSON body `sesskey` field |
| 4 | HIGH | CORS wildcard + `allow_credentials=True` | `server.py` — restricted to `127.0.0.1`/`localhost`, credentials disabled |
| 5 | HIGH | `innerHTML` on unsanitized LLM output | `chat_interface.js` — DOMPurify added to `renderMarkdown()` |
| 6 | HIGH | Plaintext API key storage (`local_craftpilot_keys`) | Table dropped; Fireworks integration removed entirely |
| 7 | MEDIUM | Unbounded user input on `/api/chat` | `api/models.py` — Pydantic `Field(max_length=...)` on all string inputs |
| 8 | MEDIUM | Video path traversal (partial) | `routes.py` — allowlist of permitted directories |
| 9 | MEDIUM | `PARAM_RAW` on message content in external API | `manage_messages.php` — changed to `PARAM_CLEANHTML` |
| 10 | LOW | Partial API key logged at startup | `config/settings.py` — logs key name only |
| 11 | LOW | User-controlled byte offset in `log_tail.php` | Clamped to `[-1, filesize]` |
| 12 | LOW | cURL error detail exposed to browser | `chat_proxy.php` — generic message only |
