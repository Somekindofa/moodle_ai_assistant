<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Admin panel for local_craftpilot: backend status, indexed courses, re-ingest, live log.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require('../../config.php');

require_login();
require_capability('moodle/site:config', context_system::instance());

$PAGE->set_url(new moodle_url('/local/craftpilot/admin_panel.php'));
$PAGE->set_context(context_system::instance());
$PAGE->set_pagelayout('admin');
$PAGE->set_title(get_string('adminpanel', 'local_craftpilot'));
$PAGE->set_heading(get_string('adminpanel', 'local_craftpilot'));
$PAGE->requires->js_call_amd('local_craftpilot/craftpilot_admin', 'init', [
    (new moodle_url('/local/craftpilot/log_tail.php'))->out(false),
    (new moodle_url('/local/craftpilot/reingest_all.php'))->out(false),
]);

// ── Backend health check ──────────────────────────────────────────────────────
$backend_healthy  = false;
$backend_message  = 'Unreachable';
$vector_doc_count = null;

$ch = curl_init('http://127.0.0.1:8000/api/health');
curl_setopt_array($ch, [CURLOPT_RETURNTRANSFER => true, CURLOPT_TIMEOUT => 3]);
$health_raw  = curl_exec($ch);
$health_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

if ($health_code === 200 && $health_raw) {
    $health_data = json_decode($health_raw, true);
    if (isset($health_data['status']) && $health_data['status'] === 'healthy') {
        $backend_healthy = true;
        $backend_message = 'healthy';
    } else {
        $backend_message = $health_raw;
    }
}

$ch = curl_init('http://127.0.0.1:8000/api/status');
curl_setopt_array($ch, [CURLOPT_RETURNTRANSFER => true, CURLOPT_TIMEOUT => 3]);
$status_raw  = curl_exec($ch);
$status_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

if ($status_code === 200 && $status_raw) {
    $status_data = json_decode($status_raw, true);
    if (isset($status_data['vector_doc_count'])) {
        $vector_doc_count = (int) $status_data['vector_doc_count'];
    } elseif (isset($status_data['total_docs'])) {
        $vector_doc_count = (int) $status_data['total_docs'];
    }
}

// ── Indexed courses table ─────────────────────────────────────────────────────
$indexed_courses = $DB->get_records_sql("
    SELECT ci.course_id,
           c.fullname  AS course_name,
           COUNT(ci.cmid) AS module_count,
           MAX(ci.last_indexed) AS last_indexed
      FROM {local_craftpilot_cm_index} ci
      LEFT JOIN {course} c ON c.id = ci.course_id
     GROUP BY ci.course_id, c.fullname
     ORDER BY ci.course_id
");

echo $OUTPUT->header();
?>

<div class="container-fluid py-3">

    <!-- Backend Status -->
    <div class="card mb-4">
        <div class="card-header fw-semibold"><?php echo get_string('backendstatus', 'local_craftpilot'); ?></div>
        <div class="card-body d-flex align-items-center gap-3">
            <?php if ($backend_healthy): ?>
                <span class="badge bg-success fs-6">&#9679; <?php echo s($backend_message); ?></span>
            <?php else: ?>
                <span class="badge bg-danger fs-6">&#9679; <?php echo s($backend_message); ?></span>
            <?php endif; ?>
            <?php if ($vector_doc_count !== null): ?>
                <span class="text-muted">
                    <?php echo get_string('vectordoccount', 'local_craftpilot'); ?>:
                    <strong><?php echo number_format($vector_doc_count); ?></strong>
                </span>
            <?php endif; ?>
        </div>
    </div>

    <!-- Indexed Courses -->
    <div class="card mb-4">
        <div class="card-header fw-semibold"><?php echo get_string('indexedcourses', 'local_craftpilot'); ?></div>
        <div class="card-body p-0">
            <?php if (empty($indexed_courses)): ?>
                <p class="p-3 text-muted mb-0">No courses indexed yet.</p>
            <?php else: ?>
                <table class="table table-sm table-striped mb-0">
                    <thead>
                        <tr>
                            <th>Course</th>
                            <th>ID</th>
                            <th>Modules</th>
                            <th><?php echo get_string('lastindexed', 'local_craftpilot'); ?></th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php foreach ($indexed_courses as $row): ?>
                            <tr>
                                <td><?php echo s($row->course_name ?: '(unknown)'); ?></td>
                                <td><?php echo (int) $row->course_id; ?></td>
                                <td><?php echo (int) $row->module_count; ?></td>
                                <td><?php echo $row->last_indexed
                                        ? userdate((int) $row->last_indexed, get_string('strftimedatetimeshort', 'langconfig'))
                                        : '—'; ?></td>
                            </tr>
                        <?php endforeach; ?>
                    </tbody>
                </table>
            <?php endif; ?>
        </div>
    </div>

    <!-- Re-ingest All -->
    <div class="card mb-4">
        <div class="card-header fw-semibold"><?php echo get_string('reingestall', 'local_craftpilot'); ?></div>
        <div class="card-body">
            <p class="text-muted mb-3"><?php echo get_string('reingestdesc', 'local_craftpilot'); ?></p>
            <button id="cp-admin-reingest-btn" class="btn btn-primary mb-3">
                <?php echo get_string('reingestbtn', 'local_craftpilot'); ?>
            </button>
            <div id="cp-admin-reingest-progress"
                 style="max-height:260px;overflow-y:auto;font-size:.85rem;font-family:monospace;"></div>
        </div>
    </div>

    <!-- Live Log -->
    <div class="card">
        <div class="card-header fw-semibold d-flex align-items-center justify-content-between">
            <span><?php echo get_string('livebackendlog', 'local_craftpilot'); ?></span>
            <span class="d-flex align-items-center gap-3">
                <label class="mb-0 d-flex align-items-center gap-1" style="font-weight:normal;cursor:pointer;">
                    <input type="checkbox" id="cp-admin-autoscroll" checked>
                    <?php echo get_string('autoscroll', 'local_craftpilot'); ?>
                </label>
                <button id="cp-admin-clear-log" class="btn btn-sm btn-outline-secondary">
                    <?php echo get_string('clearview', 'local_craftpilot'); ?>
                </button>
            </span>
        </div>
        <div class="card-body p-0">
            <pre id="cp-admin-log"
                 style="height:340px;overflow-y:auto;margin:0;padding:.75rem;font-size:.8rem;
                        background:#1e1e1e;color:#d4d4d4;border-radius:0 0 .375rem .375rem;"></pre>
        </div>
    </div>

</div>

<?php
echo $OUTPUT->footer();
