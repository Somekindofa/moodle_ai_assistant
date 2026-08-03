// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * AMD module for the CraftPilot RAG Test Bench admin page.
 *
 * Handles: run button + SSE progress, result card rendering, notes auto-save,
 * flag toggle, run history navigation, and flagged export.
 *
 * @module     local_craftpilot/test_bench
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

/** @type {{runUrl: string, ajaxUrl: string, sesskey: string, activeRun: boolean,
 *          currentRunId: number|null, noteTimers: Object}} */
const state = {
    runUrl:       null,
    ajaxUrl:      null,
    sesskey:      null,
    activeRun:    false,
    currentRunId: null,
    noteTimers:   {},
};

// ── Entry point ───────────────────────────────────────────────────────────────

/**
 * Initialise the test bench UI.
 *
 * @param {string} runUrl   URL of test_bench_run.php (SSE endpoint).
 * @param {string} ajaxUrl  URL of test_bench_ajax.php.
 * @param {string} sesskey  Moodle session key for CSRF protection.
 */
export const init = (runUrl, ajaxUrl, sesskey) => {
    state.runUrl  = runUrl;
    state.ajaxUrl = ajaxUrl;
    state.sesskey = sesskey;

    bindRunButton();
    bindExportButton();
    bindHistorySidebar();
};

// ── Run button ────────────────────────────────────────────────────────────────

function bindRunButton() {
    const btn = document.getElementById('cp-tb-run-btn');
    if (btn) {
        btn.addEventListener('click', startRun);
    }
}

function startRun() {
    if (state.activeRun) {
        return;
    }
    if (!confirm('Launch all test questions against the RAG backend?\nThis may take several minutes.')) {
        return;
    }

    state.activeRun = true;
    setStatus('running', 'Running…');
    clearResults();
    showProgress();

    // EventSource only supports GET, so sesskey goes in the query string.
    const url = state.runUrl + '?sesskey=' + encodeURIComponent(state.sesskey);
    const es  = new EventSource(url);

    es.addEventListener('run_start', (e) => {
        const d = JSON.parse(e.data);
        state.currentRunId = d.run_id;
        addProgressLine('info', `Run started — ${d.total} questions — ID: ${d.run_uuid}`);
        prependRunToHistory(d.run_id, d.run_uuid, Math.floor(Date.now() / 1000), d.total);
    });

    es.addEventListener('question_start', (e) => {
        const d = JSON.parse(e.data);
        addProgressLine('info', `[${d.id}] ${d.label} — querying backend…`);
    });

    es.addEventListener('question_done', (e) => {
        const d = JSON.parse(e.data);
        addProgressLine('success', `[${d.question_id}] ${d.question_label} — done (${d.execution_time_ms}ms)`);
        appendResultCard(d);
        hideEmptyMsg();
    });

    es.addEventListener('question_error', (e) => {
        const d = JSON.parse(e.data);
        addProgressLine('error', `[${d.id}] ERROR: ${d.message}`);
        appendErrorCard(d);
        hideEmptyMsg();
    });

    es.addEventListener('run_done', () => {
        es.close();
        state.activeRun = false;
        setStatus('done', 'Done');
        addProgressLine('info', 'All questions complete.');
    });

    es.onerror = () => {
        es.close();
        state.activeRun = false;
        setStatus('error', 'Connection lost');
        addProgressLine('error', 'SSE connection lost — the run may have completed partially.');
    };
}

// ── Result cards ──────────────────────────────────────────────────────────────

function appendResultCard(d) {
    const container = document.getElementById('cp-tb-results');
    if (!container) {
        return;
    }
    container.appendChild(buildCard(d));
}

function appendErrorCard(d) {
    const container = document.getElementById('cp-tb-results');
    if (!container) {
        return;
    }
    const card = document.createElement('div');
    card.className = 'cp-tb-qcard cp-tb-qcard--error';
    card.innerHTML =
        `<div class="cp-tb-qcard-header">` +
        `<span class="cp-tb-q-badge">${esc(d.id || '?')}</span>` +
        `<span class="cp-tb-q-label">${esc(d.label || '')}</span>` +
        `</div>` +
        `<div class="alert alert-danger mb-0 mt-2" style="font-size:.85rem;">` +
        `Backend error: ${esc(d.message)}` +
        `</div>`;
    container.appendChild(card);
}

function buildCard(d) {
    // d = {index, record_id, question_id, question_label, question_scenario,
    //      question_text, generated_text, retrieved_sources, execution_time_ms,
    //      flagged, notes, refined_query}
    const card  = document.createElement('div');
    const isErr = !d.generated_text || d.generated_text.startsWith('[ERROR:');
    card.className = 'cp-tb-qcard' + (isErr ? ' cp-tb-qcard--error' : '');
    card.dataset.resultId = d.record_id;

    const flagClass = d.flagged ? 'cp-tb-flag cp-tb-flag--active' : 'cp-tb-flag';

    card.innerHTML =
        // Header row
        `<div class="cp-tb-qcard-header">` +
            `<span class="cp-tb-q-badge">${esc(d.question_id)}</span>` +
            `<span class="cp-tb-q-label">${esc(d.question_label)}</span>` +
            `<span class="cp-tb-q-scenario text-muted">${esc(d.question_scenario || '')}</span>` +
            `<span class="cp-tb-q-time">${d.execution_time_ms}ms</span>` +
            `<button class="${flagClass}" ` +
                     `data-record-id="${d.record_id}" ` +
                     `aria-pressed="${d.flagged ? 'true' : 'false'}" ` +
                     `title="Flag for troubleshooting">&#9873;</button>` +
        `</div>` +

        // Question text
        `<div class="cp-tb-q-text">${esc(d.question_text)}</div>` +

        // Refined query (if available)
        (d.refined_query
            ? `<div class="cp-tb-refined-query"><strong>Refined query (PRF):</strong> ${esc(d.refined_query)}</div>`
            : '') +

        // Sources
        buildSourcesHtml(d.retrieved_sources) +

        // Generated text
        `<div class="cp-tb-generated">` +
            `<div class="cp-tb-generated-label">Generated response</div>` +
            `<div class="cp-tb-generated-text">${esc(d.generated_text || '')}</div>` +
        `</div>` +

        // Notes
        `<div class="cp-tb-notes-wrap">` +
            `<label class="cp-tb-notes-label" for="cp-tb-notes-${d.record_id}">Notes</label>` +
            `<textarea id="cp-tb-notes-${d.record_id}" ` +
                      `class="cp-tb-notes form-control" ` +
                      `rows="2" ` +
                      `data-record-id="${d.record_id}" ` +
                      `placeholder="Observations on this result…">${esc(d.notes || '')}</textarea>` +
            `<span class="cp-tb-saved-indicator" id="cp-tb-saved-${d.record_id}" hidden>Saved</span>` +
        `</div>`;

    card.querySelector('.cp-tb-flag').addEventListener('click', handleFlagClick);
    card.querySelector('.cp-tb-notes').addEventListener('input', handleNotesInput);

    return card;
}

function buildSourcesHtml(sources) {
    const vids = (sources && sources.videos)    ? sources.videos    : [];
    const docs = (sources && sources.documents) ? sources.documents : [];

    if (!vids.length && !docs.length) {
        return '<div class="cp-tb-sources cp-tb-sources--empty text-muted">No sources retrieved (guardrail or empty corpus).</div>';
    }

    let inner = '';

    vids.forEach((v) => {
        const name  = v.filename || v.filepath || v.video_id || '(unknown video)';
        const start = formatTime(v.start_time);
        const end   = formatTime(v.end_time);
        inner +=
            `<div class="cp-tb-src cp-tb-src--video">` +
            `<span class="badge bg-primary">Video</span>` +
            `<span class="cp-tb-src-name">${esc(name)}</span>` +
            `<span class="cp-tb-src-time text-muted">${start} – ${end}</span>` +
            `</div>`;
    });

    docs.forEach((doc) => {
        const title   = doc.module_name  || '(unknown module)';
        const heading = doc.heading_path || '';
        const snippet = (doc.content || '').substring(0, 150);
        inner +=
            `<div class="cp-tb-src cp-tb-src--doc">` +
            `<span class="badge bg-secondary">Course</span>` +
            `<span class="cp-tb-src-name">${esc(title)}</span>` +
            (heading ? `<span class="cp-tb-src-path text-muted">${esc(heading)}</span>` : '') +
            `<span class="cp-tb-src-snippet text-muted">${esc(snippet)}${snippet.length >= 150 ? '…' : ''}</span>` +
            `</div>`;
    });

    const count = vids.length + docs.length;
    return (
        `<details class="cp-tb-sources">` +
        `<summary>Sources (${count})</summary>` +
        `<div class="cp-tb-sources-body">${inner}</div>` +
        `</details>`
    );
}

// ── Flag toggle ───────────────────────────────────────────────────────────────

function handleFlagClick(e) {
    const btn      = e.currentTarget;
    const recordId = parseInt(btn.dataset.recordId, 10);

    fetch(state.ajaxUrl, {
        method:  'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body:    new URLSearchParams({
            action:    'toggleflag',
            sesskey:   state.sesskey,
            record_id: recordId,
        }),
    })
    .then((r) => r.json())
    .then((d) => {
        if (!d.ok) {
            return;
        }
        const active = !!d.flagged;
        btn.classList.toggle('cp-tb-flag--active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        updateHistoryFlagCount(d.run_id, d.flagged_count);
    })
    .catch(() => { /* ignore network errors on flag */ });
}

// ── Notes auto-save (debounced 600 ms) ────────────────────────────────────────

function handleNotesInput(e) {
    const ta       = e.currentTarget;
    const recordId = parseInt(ta.dataset.recordId, 10);

    if (state.noteTimers[recordId]) {
        clearTimeout(state.noteTimers[recordId]);
    }

    state.noteTimers[recordId] = setTimeout(() => {
        fetch(state.ajaxUrl, {
            method:  'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body:    new URLSearchParams({
                action:    'savenotes',
                sesskey:   state.sesskey,
                record_id: recordId,
                notes:     ta.value,
            }),
        })
        .then((r) => r.json())
        .then((d) => {
            if (d.ok) {
                const indicator = document.getElementById('cp-tb-saved-' + recordId);
                if (indicator) {
                    indicator.hidden = false;
                    setTimeout(() => { indicator.hidden = true; }, 1500);
                }
            }
        })
        .catch(() => { /* ignore */ });
    }, 600);
}

// ── History sidebar ───────────────────────────────────────────────────────────

function bindHistorySidebar() {
    const sidebar = document.getElementById('cp-tb-history');
    if (!sidebar) {
        return;
    }
    sidebar.addEventListener('click', (e) => {
        const li = e.target.closest('[data-run-id]');
        if (!li) {
            return;
        }
        loadRun(parseInt(li.dataset.runId, 10));
    });
}

function loadRun(runId) {
    clearResults();
    setStatus('loading', 'Loading…');

    fetch(state.ajaxUrl, {
        method:  'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body:    new URLSearchParams({
            action:  'loadrun',
            sesskey: state.sesskey,
            run_id:  runId,
        }),
    })
    .then((r) => r.json())
    .then((d) => {
        if (!d.ok) {
            setStatus('error', 'Load failed');
            return;
        }
        state.currentRunId = runId;
        hideEmptyMsg();

        d.results.forEach((r) => {
            const qidx = r.question_index;
            appendResultCard({
                index:              qidx,
                record_id:          r.id,
                question_id:        'Q' + String(qidx + 1).padStart(2, '0'),
                question_label:     '',
                question_scenario:  '',
                question_text:      r.question_text,
                generated_text:     r.generated_text,
                retrieved_sources:  r.retrieved_sources,
                refined_query:      r.refined_query,
                execution_time_ms:  r.execution_time_ms,
                flagged:            r.flagged,
                notes:              r.notes,
            });
        });

        setStatus('done', 'Loaded');
        highlightHistoryItem(runId);
    })
    .catch(() => setStatus('error', 'Load failed'));
}

function prependRunToHistory(runId, runUuid, createdTime, total) {
    const list = document.getElementById('cp-tb-history-list');
    if (!list) {
        // Sidebar has no list element yet — create one.
        const sidebar = document.getElementById('cp-tb-history');
        const body    = sidebar && sidebar.querySelector('.card-body');
        if (!body) {
            return;
        }
        // Remove "No runs yet" paragraph.
        const placeholder = body.querySelector('p');
        if (placeholder) {
            placeholder.remove();
        }
        const ul = document.createElement('ul');
        ul.id = 'cp-tb-history-list';
        ul.className = 'mb-0';
        body.appendChild(ul);
    }

    const ul = document.getElementById('cp-tb-history-list');
    if (!ul) {
        return;
    }

    const d   = new Date(createdTime * 1000);
    const fmt = d.toLocaleString('fr-FR', {day:'2-digit', month:'short', year:'numeric',
                                           hour:'2-digit', minute:'2-digit'});
    const li  = document.createElement('li');
    li.dataset.runId = runId;
    li.innerHTML =
        `<div class="cp-tb-hist-date">${fmt}</div>` +
        `<div class="cp-tb-hist-meta">` +
            `${total} questions ` +
            `<span class="cp-tb-hist-flagged" id="cp-tb-hist-flag-${runId}" style="display:none;">` +
            `&#9873; 0</span>` +
        `</div>`;

    ul.insertBefore(li, ul.firstChild);
}

function highlightHistoryItem(runId) {
    document.querySelectorAll('#cp-tb-history-list li').forEach((li) => {
        li.classList.toggle('active', parseInt(li.dataset.runId, 10) === runId);
    });
}

function updateHistoryFlagCount(runId, count) {
    const el = document.getElementById('cp-tb-hist-flag-' + runId);
    if (!el) {
        return;
    }
    el.textContent = '\u2691 ' + count;
    el.style.display = count > 0 ? '' : 'none';
}

// ── Export flagged ────────────────────────────────────────────────────────────

function bindExportButton() {
    const btn = document.getElementById('cp-tb-export-btn');
    if (btn) {
        btn.addEventListener('click', exportFlagged);
    }
}

function exportFlagged() {
    fetch(state.ajaxUrl, {
        method:  'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body:    new URLSearchParams({
            action:  'exportflagged',
            sesskey: state.sesskey,
        }),
    })
    .then((r) => r.json())
    .then((data) => {
        if (!data.test_runs || !data.test_runs.length) {
            alert('No flagged results to export.');
            return;
        }
        const blob     = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
        const url      = URL.createObjectURL(blob);
        const anchor   = document.createElement('a');
        anchor.href    = url;
        anchor.download = 'craftpilot_flagged_' + Date.now() + '.json';
        anchor.click();
        URL.revokeObjectURL(url);
    })
    .catch(() => alert('Export failed — check console.'));
}

// ── Progress console ──────────────────────────────────────────────────────────

function showProgress() {
    const el = document.getElementById('cp-tb-progress');
    if (el) {
        el.style.display = '';
        el.innerHTML = '';
    }
}

function addProgressLine(type, message) {
    const el = document.getElementById('cp-tb-progress');
    if (!el) {
        return;
    }
    const colors = {info: '#9cdcfe', success: '#4ec994', error: '#f44747'};
    const color  = colors[type] || '#d4d4d4';
    const line   = document.createElement('div');
    line.style.color = color;
    line.textContent = message;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
}

// ── Misc helpers ──────────────────────────────────────────────────────────────

function clearResults() {
    const el = document.getElementById('cp-tb-results');
    if (el) {
        el.innerHTML = '<p id="cp-tb-empty-msg" class="text-muted" style="display:none;"></p>';
    }
    const progress = document.getElementById('cp-tb-progress');
    if (progress) {
        progress.style.display = 'none';
        progress.innerHTML = '';
    }
}

function hideEmptyMsg() {
    const msg = document.getElementById('cp-tb-empty-msg');
    if (msg) {
        msg.style.display = 'none';
    }
}

function setStatus(type, text) {
    const el = document.getElementById('cp-tb-status');
    if (!el) {
        return;
    }
    el.style.display = '';
    el.textContent   = text;
    el.className     = 'badge ms-auto';
    const map = {running: 'bg-warning text-dark', done: 'bg-success',
                 error: 'bg-danger', loading: 'bg-info text-dark'};
    el.classList.add(...(map[type] || 'bg-secondary').split(' '));
}

function esc(str) {
    return String(str ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function formatTime(ms) {
    if (ms === null || ms === undefined) {
        return '—';
    }
    const s = ms / 1000;
    if (s < 60) {
        return s.toFixed(1) + 's';
    }
    return Math.floor(s / 60) + 'm' + String(Math.floor(s % 60)).padStart(2, '0') + 's';
}
