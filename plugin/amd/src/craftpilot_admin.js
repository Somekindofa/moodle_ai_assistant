// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Admin panel AMD module: live log viewer + re-ingest EventSource client.
 *
 * @module     local_craftpilot/craftpilot_admin
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

/**
 * Polls log_tail.php every 2 s and appends new lines to the log panel.
 *
 * @param {string} url - URL of log_tail.php
 */
function initLogViewer(url) {
    let offset = -1; // -1 → server starts at last 4 KB
    const logEl  = document.getElementById('cp-admin-log');
    const autoEl = document.getElementById('cp-admin-autoscroll');

    document.getElementById('cp-admin-clear-log').addEventListener('click', () => {
        logEl.textContent = '';
    });

    setInterval(() => {
        fetch(url + '?offset=' + offset)
            .then(r => r.json())
            .then(d => {
                if (d.lines && d.lines.length) {
                    logEl.textContent += d.lines.join('\n');
                    if (autoEl.checked) {
                        logEl.scrollTop = logEl.scrollHeight;
                    }
                }
                offset = d.offset;
            })
            .catch(() => { /* silent on network error */ });
    }, 2000);
}

/**
 * Wires the re-ingest button to an EventSource stream from reingest_all.php.
 *
 * @param {string} url - URL of reingest_all.php
 */
function initReingestButton(url) {
    const btn      = document.getElementById('cp-admin-reingest-btn');
    const progress = document.getElementById('cp-admin-reingest-progress');

    btn.addEventListener('click', () => {
        // eslint-disable-next-line no-alert
        if (!confirm('Re-ingest all course modules into ChromaDB? This may take several minutes.')) {
            return;
        }
        btn.disabled   = true;
        progress.innerHTML = '';

        // EventSource only supports GET; pass sesskey as query param.
        const es = new EventSource(url + '?sesskey=' + M.cfg.sesskey);

        es.addEventListener('progress', e => {
            const d   = JSON.parse(e.data);
            const cls = d.type === 'error' ? 'text-danger'
                      : d.type === 'info'  ? 'text-muted'
                      :                      'text-success';

            // Built with DOM APIs, never innerHTML: d.message embeds
            // teacher-controlled course and module names (see
            // reingest_all.php), which would otherwise execute as markup
            // in the site administrator's browser.
            const row = document.createElement('div');
            row.className = cls;
            if (d.done != null && d.total != null) {
                const counter = document.createElement('span');
                counter.className   = 'text-muted';
                counter.textContent = `${d.done}/${d.total} `;
                row.appendChild(counter);
            }
            row.appendChild(document.createTextNode(d.message ?? ''));
            progress.appendChild(row);
            progress.scrollTop = progress.scrollHeight;
        });

        es.addEventListener('done', e => {
            const d = JSON.parse(e.data);
            es.close();
            btn.disabled = false;

            const summary = document.createElement('div');
            summary.className   = 'alert alert-success mt-2';
            summary.textContent =
                `Done — ${d.done} indexed, ${d.skipped} skipped, ${d.errors} errors.`;
            progress.appendChild(summary);
            progress.scrollTop = progress.scrollHeight;
        });

        es.onerror = () => {
            es.close();
            btn.disabled = false;

            const err = document.createElement('div');
            err.className   = 'alert alert-danger mt-2';
            err.textContent = 'Connection lost.';
            progress.appendChild(err);
        };
    });
}

/**
 * Module entry point called by Moodle AMD loader.
 *
 * @param {string} logTailUrl  - URL for log_tail.php
 * @param {string} reingestUrl - URL for reingest_all.php
 */
export const init = (logTailUrl, reingestUrl) => {
    initLogViewer(logTailUrl);
    initReingestButton(reingestUrl);
};
