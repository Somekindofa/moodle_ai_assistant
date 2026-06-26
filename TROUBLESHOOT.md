# Troubleshooting Guide

This file documents known failure modes, their symptoms, diagnosis steps, and fixes.

---

## 1. Moodle admin JavaScript completely broken ("More" button unresponsive, dropdowns dead)

**Symptom:** All JavaScript on Moodle pages stops working. The secondary navigation "More" dropdown doesn't respond to clicks. Any JS-driven UI (modals, AJAX, etc.) is broken sitewide.

**Browser console error:**
```
/lib/requirejs.php/...core/first.js:NNNNN Uncaught SyntaxError: Cannot use import statement outside a module
require.min.js:5 Uncaught Error: No define call for core/first
```

**Root cause:** Moodle's RequireJS AMD loader bundles `local_craftpilot/chat_interface` into `core/first.js`. If `amd/build/chat_interface.min.js` contains raw ES6 `import` statements (i.e. the Babel build didn't run or failed silently), RequireJS crashes on every page load.

The `.min.js` file being identical in size to the source `.js` is the tell: Babel produced no output, or the source was copied over the build.

**Diagnosis:**
```bash
# Check if source and build are identical (they should NOT be)
diff /var/www/html/public/local/craftpilot/amd/src/chat_interface.js \
     /var/www/html/public/local/craftpilot/amd/build/chat_interface.min.js
# If output is empty → files are identical → Babel build is broken

# Check the build file starts with define() not import
head -3 /var/www/html/public/local/craftpilot/amd/build/chat_interface.min.js

# Verify the live bundle served to the browser is clean
curl -sk "https://aimove.minesparis.psl.eu/lib/requirejs.php/$(grep jsrev <<< "$(curl -sk https://aimove.minesparis.psl.eu/)" | grep -o '[0-9]\{10\}' | head -1)/core/first.js" | grep -c "^import"
# Should return 0
```

**Fix:**
```bash
# 1. Re-run the Babel build
cd /var/www/html/public/local/craftpilot
node_modules/.bin/grunt babel:dist

# 2. Verify the build now starts with define()
head -2 amd/build/chat_interface.min.js
# Expected: define(["exports", "core/ajax", ...], function(...) {

# 3. Purge Moodle localcache so the bundle is regenerated
find /var/www/moodledata/localcache -name "*.js" -delete

# 4. Hard-refresh the browser (Ctrl+Shift+R)
```

**After any edit to chat_interface.js, always re-run the Babel build before testing in the browser.**

---

## 2. Moodle cache purge (when admin UI works)

**Via admin UI:** Site administration → Development → Purge all caches

**Via CLI (when UI is broken, ask user to run — no sudo access):**
```
! sudo -u apache php /var/www/html/admin/cli/purge_caches.php
```

**Via filesystem (purge only JS localcache):**
```bash
find /var/www/moodledata/localcache -name "*.js" -delete
```

---

## 3. Large video uploads failing (disk full / temp overflow)

**Symptom:** Video upload fails partway through; error may mention disk space or temp file.

**Root cause:** Python's `tempfile` defaults to `/tmp` which is a dedicated 2 GB filesystem. Large video uploads spooled to `/tmp` overflow it.

**Fix (already applied in backend/main.py):** At startup, temp dir is redirected to `/var/video_uploads/.tmp` (198 GB partition):
```python
tmp_override = Path(os.getenv("TMPDIR_OVERRIDE", "/var/video_uploads/.tmp"))
tmp_override.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(tmp_override)
os.environ["TMPDIR"] = str(tmp_override)
```

**Diagnosis:**
```bash
df -h /tmp /var/video_uploads
# /tmp should show ~2 GB total; /var is 198 GB
```

---

## 4. SSL certificate expired (aimove.minesparis.psl.eu)

**Symptom:** Browser shows `ERR_CERT_DATE_INVALID`. Playwright/automation tools can't reach the site.

**Certificate info:**
- Type: Sectigo wildcard (`*.minesparis.psl.eu`)
- Files: `/etc/httpd/certs/wildcard.crt`, `/etc/httpd/certs/wildcard.key`, `/etc/httpd/certs/sectigo.crt`
- Expired: March 13, 2024 — renewal must go through IT/Mines Paris sysadmin team

**Workaround (to access admin without renewing):** In Chrome/Edge click "Advanced → Proceed"; in Firefox "Accept the Risk and Continue".

**Diagnosis:**
```bash
echo | openssl s_client -connect aimove.minesparis.psl.eu:443 -servername aimove.minesparis.psl.eu 2>/dev/null \
  | openssl x509 -noout -dates
```

**To reach the site from scripts (ignore cert):**
```bash
curl -sk https://aimove.minesparis.psl.eu/...
```

---

## 5. Plugin version mismatch blocking Moodle admin

**Symptom:** Moodle shows "upgrade required" banner or blocks access to admin after plugin file changes.

**Diagnosis:**
```bash
# Compare version.php vs DB
grep 'version' /var/www/html/public/local/videoelicit/version.php
mysql -u moodleuser -p<DB_PASS> -h localhost moodle \
  -e "SELECT plugin, name, value FROM mdl_config_plugins WHERE plugin='local_videoelicit' AND name='version';"
```

**Fix:** Version must match. Either bump `version.php` and run the Moodle upgrade, or revert the version number to match the DB value.

---

## 6. Key file locations

| What | Path |
|---|---|
| CraftPilot AMD source | `/var/www/html/public/local/craftpilot/amd/src/` |
| CraftPilot AMD build | `/var/www/html/public/local/craftpilot/amd/build/` |
| CraftPilot Gruntfile | `/var/www/html/public/local/craftpilot/Gruntfile.js` |
| VideoElicit Moodle plugin | `/var/www/html/public/local/videoelicit/` |
| Moodle localcache | `/var/www/moodledata/localcache/` |
| Moodle dataroot | `/var/www/moodledata/` |
| Moodle config | `/var/www/html/config.php` |
| SSL certs | `/etc/httpd/certs/` |
| CraftPilot backend log | `/tmp/craftpilot_backend.log` |
| Video upload temp dir | `/var/video_uploads/.tmp` |
| CLI cache purge | `/var/www/html/admin/cli/purge_caches.php` |
