<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * SSE endpoint: re-ingests all supported course modules into ChromaDB.
 *
 * Streams Server-Sent Events with progress updates.
 * Requires moodle/site:config capability + valid sesskey.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

// SSE responses must not be treated as AJAX (no JSON error wrapper).
define('NO_MOODLE_COOKIES', false);

require('../../config.php');

require_login();
require_capability('moodle/site:config', context_system::instance());
require_sesskey();

// ── SSE headers ───────────────────────────────────────────────────────────────
header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');
header('X-Accel-Buffering: no');   // disable nginx / Apache mod_deflate buffering
header('Connection: keep-alive');

// Disable output buffering as thoroughly as possible.
while (ob_get_level()) {
    ob_end_flush();
}

set_time_limit(300);

// ── SSE helper ────────────────────────────────────────────────────────────────
function sse(string $event, array $data): void {
    echo "event: {$event}\ndata: " . json_encode($data, JSON_UNESCAPED_UNICODE) . "\n\n";
    flush();
}

// ── Step 1: clear the index table ────────────────────────────────────────────
try {
    $DB->delete_records('local_craftpilot_cm_index');
} catch (\Throwable $e) {
    sse('progress', ['type' => 'error', 'message' => 'Failed to clear index: ' . $e->getMessage()]);
    sse('done', ['done' => 0, 'skipped' => 0, 'errors' => 1, 'total' => 0]);
    exit;
}

sse('progress', ['type' => 'info', 'message' => 'Cleared local index table.']);

// ── Step 2: collect all supported modules ─────────────────────────────────────
$supported_types = ['page', 'label', 'resource'];
$placeholders    = implode(',', array_fill(0, count($supported_types), '?'));

$modules = $DB->get_records_sql("
    SELECT cm.id AS cmid,
           cm.course AS course_id,
           m.name   AS modname,
           c.fullname AS course_name
      FROM {course_modules} cm
      JOIN {modules} m ON m.id = cm.module
      JOIN {course} c  ON c.id = cm.course
     WHERE m.name IN ($placeholders)
       AND cm.deletioninprogress = 0
     ORDER BY cm.course, cm.id
", $supported_types);

$total   = count($modules);
$done    = 0;
$skipped = 0;
$errors  = 0;

sse('progress', ['type' => 'info', 'message' => "Found {$total} modules to process."]);

// ── Step 3: process each module ───────────────────────────────────────────────
$extractor = new \local_craftpilot\course_content_extractor();
$client    = new \local_craftpilot\backend_client();

foreach ($modules as $mod) {
    $cmid      = (int) $mod->cmid;
    $course_id = (int) $mod->course_id;
    $modname   = $mod->modname;

    // Extract content.
    try {
        $payload = $extractor->extract_module($cmid, $modname, $course_id);
    } catch (\Throwable $e) {
        $errors++;
        sse('progress', [
            'done'    => $done,
            'total'   => $total,
            'type'    => 'error',
            'message' => "cmid={$cmid} ({$modname}): extract failed — " . $e->getMessage(),
        ]);
        continue;
    }

    if (empty($payload)) {
        $skipped++;
        sse('progress', [
            'done'    => $done,
            'total'   => $total,
            'type'    => 'info',
            'message' => "cmid={$cmid} ({$modname}): skipped (no extractable content).",
        ]);
        continue;
    }

    // Delete existing ChromaDB chunks (safe if already empty).
    try {
        $client->delete_module($course_id, $cmid);
    } catch (\Throwable $e) {
        // Non-fatal: log but continue.
        error_log("CraftPilot reingest: delete failed for cmid={$cmid}: " . $e->getMessage());
    }

    // Ingest into ChromaDB.
    try {
        $client->ingest_module($course_id, $cmid, $modname, $payload);
    } catch (\Throwable $e) {
        $errors++;
        sse('progress', [
            'done'    => $done,
            'total'   => $total,
            'type'    => 'error',
            'message' => "cmid={$cmid} ({$modname}): ingest failed — " . $e->getMessage(),
        ]);
        continue;
    }

    // Record in the local index table.
    $existing = $DB->get_record('local_craftpilot_cm_index', ['cmid' => $cmid]);
    $now      = time();
    $md5      = isset($payload['content_html']) ? md5($payload['content_html']) : md5(json_encode($payload));

    if ($existing) {
        $DB->update_record('local_craftpilot_cm_index', (object)[
            'id'           => $existing->id,
            'content_hash' => $md5,
            'last_indexed' => $now,
        ]);
    } else {
        $DB->insert_record('local_craftpilot_cm_index', (object)[
            'cmid'         => $cmid,
            'course_id'    => $course_id,
            'content_hash' => $md5,
            'last_indexed' => $now,
        ]);
    }

    $done++;
    sse('progress', [
        'done'        => $done,
        'total'       => $total,
        'type'        => 'success',
        'message'     => "cmid={$cmid} ({$modname}) in \"{$mod->course_name}\" — OK",
        'course_name' => $mod->course_name,
        'modname'     => $modname,
        'cmid'        => $cmid,
    ]);
}

// ── Step 4: done ─────────────────────────────────────────────────────────────
sse('done', [
    'done'    => $done,
    'skipped' => $skipped,
    'errors'  => $errors,
    'total'   => $total,
]);
