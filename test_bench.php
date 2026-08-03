<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * RAG Test Bench admin page for local_craftpilot.
 *
 * Provides a UI to run MOCO 2026 evaluation questions against the RAG pipeline,
 * browse past runs, annotate results, flag issues, and export for LLM troubleshooting.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require('../../config.php');

require_login();
require_capability('moodle/site:config', context_system::instance());

$PAGE->set_url(new moodle_url('/local/craftpilot/test_bench.php'));
$PAGE->set_context(context_system::instance());
$PAGE->set_pagelayout('admin');
$PAGE->set_title(get_string('testbench', 'local_craftpilot'));
$PAGE->set_heading(get_string('testbench', 'local_craftpilot'));

$PAGE->requires->js_call_amd('local_craftpilot/test_bench', 'init', [
    (new moodle_url('/local/craftpilot/test_bench_run.php'))->out(false),
    (new moodle_url('/local/craftpilot/test_bench_ajax.php'))->out(false),
    sesskey(),
]);

// ── Load recent run history for sidebar (last 30 runs) ───────────────────────
$past_runs = $DB->get_records(
    'local_craftpilot_testrun',
    null,
    'created_time DESC',
    'id, run_uuid, created_time, question_count, flagged_count',
    0, 30
);

// ── Load question list for the "Questions" panel ──────────────────────────────
require_once(__DIR__ . '/classes/test_bench_questions.php');

echo $OUTPUT->header();
?>

<div class="container-fluid py-3" id="cp-tb-layout">

    <!-- ── History sidebar ─────────────────────────────────────────────────── -->
    <aside id="cp-tb-history" class="card">
        <div class="card-header fw-semibold">
            <?php echo get_string('testrunhistory', 'local_craftpilot'); ?>
        </div>
        <div class="card-body p-0">
            <?php if (empty($past_runs)): ?>
                <p class="p-3 text-muted mb-0" style="font-size:.85rem;">No runs yet.</p>
            <?php else: ?>
                <ul id="cp-tb-history-list" class="mb-0">
                    <?php foreach ($past_runs as $run): ?>
                        <li data-run-id="<?php echo (int) $run->id; ?>">
                            <div class="cp-tb-hist-date">
                                <?php echo userdate((int) $run->created_time, '%d %b %Y %H:%M'); ?>
                            </div>
                            <div class="cp-tb-hist-meta">
                                <?php echo (int) $run->question_count; ?> questions
                                <?php if ($run->flagged_count > 0): ?>
                                    <span class="cp-tb-hist-flagged"
                                          id="cp-tb-hist-flag-<?php echo (int) $run->id; ?>">
                                        &#9873; <?php echo (int) $run->flagged_count; ?>
                                    </span>
                                <?php else: ?>
                                    <span class="cp-tb-hist-flagged"
                                          id="cp-tb-hist-flag-<?php echo (int) $run->id; ?>"
                                          style="display:none;">
                                        &#9873; 0
                                    </span>
                                <?php endif; ?>
                            </div>
                        </li>
                    <?php endforeach; ?>
                </ul>
            <?php endif; ?>
        </div>
    </aside>

    <!-- ── Main panel ──────────────────────────────────────────────────────── -->
    <main id="cp-tb-main">

        <!-- Header toolbar -->
        <div class="d-flex align-items-center gap-3 mb-3 flex-wrap">
            <h2 class="mb-0 h5"><?php echo get_string('testbench', 'local_craftpilot'); ?></h2>
            <button id="cp-tb-run-btn" class="btn btn-primary">
                &#9654; <?php echo get_string('runtests', 'local_craftpilot'); ?>
            </button>
            <button id="cp-tb-export-btn" class="btn btn-outline-secondary">
                &#8659; <?php echo get_string('exportflagged', 'local_craftpilot'); ?>
            </button>
            <span id="cp-tb-status" class="badge bg-secondary ms-auto" style="display:none;"></span>
        </div>

        <!-- Questions reference panel (collapsed by default) -->
        <details class="card mb-3">
            <summary class="card-header fw-semibold" style="cursor:pointer;list-style:none;">
                <?php echo get_string('testbenchquestions', 'local_craftpilot'); ?>
                (<?php echo count(TESTBENCH_QUESTIONS); ?>)
            </summary>
            <div class="card-body p-0">
                <table class="table table-sm table-striped mb-0">
                    <thead>
                        <tr>
                            <th style="width:50px;">ID</th>
                            <th style="width:200px;"><?php echo get_string('tbscenario', 'local_craftpilot'); ?></th>
                            <th>Question</th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php foreach (TESTBENCH_QUESTIONS as $q): ?>
                            <tr>
                                <td><span class="badge bg-secondary"><?php echo s($q['id']); ?></span></td>
                                <td><small class="text-muted"><?php echo s($q['label']); ?></small></td>
                                <td><small><?php echo s($q['text']); ?></small></td>
                            </tr>
                        <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </details>

        <!-- Progress console (hidden until a run starts) -->
        <div id="cp-tb-progress" style="display:none;"></div>

        <!-- Results container -->
        <div id="cp-tb-results">
            <p id="cp-tb-empty-msg" class="text-muted">
                <?php echo get_string('notestresults', 'local_craftpilot'); ?>
            </p>
        </div>

    </main>

</div>

<?php
echo $OUTPUT->footer();
