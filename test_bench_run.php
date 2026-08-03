<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * SSE endpoint: runs all MOCO 2026 test questions against the RAG backend,
 * persists results to the DB, and streams per-question progress events.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

define('NO_MOODLE_COOKIES', false);

require('../../config.php');

require_login();
require_capability('moodle/site:config', context_system::instance());
require_sesskey();

// ── SSE headers ───────────────────────────────────────────────────────────────
header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');
header('X-Accel-Buffering: no');
header('Connection: keep-alive');

while (ob_get_level()) {
    ob_end_flush();
}

set_time_limit(600);  // 8 questions × up to ~60s each, plus margin.

// ── SSE helper ────────────────────────────────────────────────────────────────
function sse(string $event, array $data): void {
    echo "event: {$event}\ndata: " . json_encode($data, JSON_UNESCAPED_UNICODE) . "\n\n";
    flush();
}

// ── Load question list ────────────────────────────────────────────────────────
require_once(__DIR__ . '/classes/test_bench_questions.php');

// ── Generate a version-4 UUID for this run ────────────────────────────────────
$run_uuid = sprintf(
    '%04x%04x-%04x-%04x-%04x-%04x%04x%04x',
    mt_rand(0, 0xffff), mt_rand(0, 0xffff),
    mt_rand(0, 0xffff),
    mt_rand(0, 0x0fff) | 0x4000,
    mt_rand(0, 0x3fff) | 0x8000,
    mt_rand(0, 0xffff), mt_rand(0, 0xffff), mt_rand(0, 0xffff)
);

// ── Insert the testrun record ─────────────────────────────────────────────────
$run_id = $DB->insert_record('local_craftpilot_testrun', (object)[
    'run_uuid'       => $run_uuid,
    'created_time'   => time(),
    'question_count' => count(TESTBENCH_QUESTIONS),
    'flagged_count'  => 0,
    'notes'          => '',
]);

sse('run_start', [
    'run_id'   => $run_id,
    'run_uuid' => $run_uuid,
    'total'    => count(TESTBENCH_QUESTIONS),
]);

// ── Run each question ─────────────────────────────────────────────────────────
$client = new \local_craftpilot\backend_client();

foreach (TESTBENCH_QUESTIONS as $idx => $q) {
    sse('question_start', [
        'index' => $idx,
        'id'    => $q['id'],
        'label' => $q['label'],
    ]);

    // Each question gets a unique conversation thread so PRF state cannot leak.
    $conv_id = $run_uuid . '-q' . $idx;

    try {
        $result = $client->chat_request_full(
            $q['text'],
            $conv_id,
            null,   // no domain filter — evaluate full cross-domain retrieval
            null    // no course filter
        );

        $record_id = $DB->insert_record('local_craftpilot_testresult', (object)[
            'run_id'            => $run_id,
            'question_index'    => $idx,
            'question_text'     => $q['text'],
            'generated_text'    => $result['generated_text'],
            'retrieved_sources' => json_encode($result['retrieved_sources'], JSON_UNESCAPED_UNICODE),
            'refined_query'     => $result['refined_query'],
            'execution_time_ms' => $result['execution_time_ms'],
            'flagged'           => 0,
            'notes'             => '',
        ]);

        sse('question_done', [
            'index'             => $idx,
            'record_id'         => $record_id,
            'question_id'       => $q['id'],
            'question_label'    => $q['label'],
            'question_scenario' => $q['scenario'],
            'question_text'     => $q['text'],
            'generated_text'    => $result['generated_text'],
            'retrieved_sources' => $result['retrieved_sources'],
            'refined_query'     => $result['refined_query'],
            'execution_time_ms' => $result['execution_time_ms'],
            'flagged'           => 0,
            'notes'             => '',
        ]);

    } catch (\Throwable $e) {
        // Insert a placeholder record so the run stays consistent.
        $record_id = $DB->insert_record('local_craftpilot_testresult', (object)[
            'run_id'            => $run_id,
            'question_index'    => $idx,
            'question_text'     => $q['text'],
            'generated_text'    => '[ERROR: ' . $e->getMessage() . ']',
            'retrieved_sources' => json_encode(['videos' => [], 'documents' => []]),
            'refined_query'     => null,
            'execution_time_ms' => 0,
            'flagged'           => 0,
            'notes'             => '',
        ]);

        sse('question_error', [
            'index'     => $idx,
            'record_id' => $record_id,
            'id'        => $q['id'],
            'label'     => $q['label'],
            'message'   => $e->getMessage(),
        ]);
    }
}

// ── Done ──────────────────────────────────────────────────────────────────────
sse('run_done', [
    'run_id'   => $run_id,
    'run_uuid' => $run_uuid,
]);
