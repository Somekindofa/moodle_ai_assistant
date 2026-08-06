<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * AJAX endpoint for the RAG test bench.
 *
 * Actions: savenotes, toggleflag, loadrun, exportflagged
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

define('AJAX_SCRIPT', true);

require('../../config.php');

require_login();
require_capability('moodle/site:config', context_system::instance());

$action = required_param('action', PARAM_ALPHA);

// All mutations require a valid sesskey.
if (!confirm_sesskey()) {
    http_response_code(403);
    echo json_encode(['error' => 'Invalid sesskey']);
    exit;
}

header('Content-Type: application/json; charset=utf-8');

switch ($action) {

    // ── savenotes ─────────────────────────────────────────────────────────────
    // Params: record_id (int, optional) OR run_id (int, optional), notes (text)
    case 'savenotes':
        $notes     = optional_param('notes',     '',  PARAM_RAW);
        $record_id = optional_param('record_id', 0,   PARAM_INT);
        $run_id    = optional_param('run_id',    0,   PARAM_INT);

        if ($record_id > 0) {
            $DB->set_field('local_craftpilot_testresult', 'notes', $notes, ['id' => $record_id]);
        } elseif ($run_id > 0) {
            $DB->set_field('local_craftpilot_testrun', 'notes', $notes, ['id' => $run_id]);
        } else {
            http_response_code(400);
            echo json_encode(['error' => 'record_id or run_id required']);
            exit;
        }

        echo json_encode(['ok' => true]);
        break;

    // ── toggleflag ────────────────────────────────────────────────────────────
    // Params: record_id (int)
    // Toggles the flagged field, recomputes and caches flagged_count on parent run.
    case 'toggleflag':
        $record_id = required_param('record_id', PARAM_INT);

        $rec = $DB->get_record('local_craftpilot_testresult', ['id' => $record_id], '*', MUST_EXIST);
        $new_flag = $rec->flagged ? 0 : 1;

        $DB->set_field('local_craftpilot_testresult', 'flagged', $new_flag, ['id' => $record_id]);

        // Recompute and cache the flagged_count on the parent testrun row.
        $flagged_count = $DB->count_records('local_craftpilot_testresult', [
            'run_id'  => $rec->run_id,
            'flagged' => 1,
        ]);
        $DB->set_field('local_craftpilot_testrun', 'flagged_count', $flagged_count, ['id' => $rec->run_id]);

        echo json_encode([
            'ok'            => true,
            'flagged'       => $new_flag,
            'flagged_count' => $flagged_count,
            'run_id'        => (int) $rec->run_id,
        ]);
        break;

    // ── loadrun ───────────────────────────────────────────────────────────────
    // Params: run_id (int)
    // Returns all result records for the given run, with retrieved_sources decoded.
    case 'loadrun':
        $run_id = required_param('run_id', PARAM_INT);

        $run = $DB->get_record('local_craftpilot_testrun', ['id' => $run_id], '*', MUST_EXIST);

        $results = $DB->get_records(
            'local_craftpilot_testresult',
            ['run_id' => $run_id],
            'question_index ASC'
        );

        foreach ($results as $r) {
            $r->retrieved_sources = json_decode($r->retrieved_sources ?? '{}', true)
                ?: ['videos' => [], 'documents' => []];
            $r->execution_time_ms = (int) $r->execution_time_ms;
            $r->flagged           = (int) $r->flagged;
            $r->question_index    = (int) $r->question_index;
        }

        echo json_encode([
            'ok'      => true,
            'run'     => $run,
            'results' => array_values($results),
        ], JSON_UNESCAPED_UNICODE);
        break;

    // ── exportflagged ─────────────────────────────────────────────────────────
    // Exports all runs that contain at least one flagged result.
    // Only flagged question results are included in the output.
    case 'exportflagged':
        $runs = $DB->get_records_sql("
            SELECT DISTINCT tr.id, tr.run_uuid, tr.created_time,
                            tr.question_count, tr.flagged_count
              FROM {local_craftpilot_testrun} tr
              JOIN {local_craftpilot_testresult} res ON res.run_id = tr.id
             WHERE res.flagged = 1
             ORDER BY tr.created_time DESC
        ");

        $export = [
            'export_timestamp' => gmdate('Y-m-d\TH:i:s\Z'),
            'export_version'   => '1.0',
            'flagged_only'     => true,
            'test_runs'        => [],
        ];

        foreach ($runs as $run) {
            $flagged_results = $DB->get_records(
                'local_craftpilot_testresult',
                ['run_id' => $run->id, 'flagged' => 1],
                'question_index ASC'
            );

            $questions = [];
            foreach ($flagged_results as $res) {
                $qidx        = (int) $res->question_index;
                $questions[] = [
                    'question_index'    => $qidx,
                    'question_id'       => 'Q' . str_pad($qidx + 1, 2, '0', STR_PAD_LEFT),
                    'question_text'     => $res->question_text,
                    'flagged'           => true,
                    'generated_text'    => $res->generated_text,
                    'execution_time_ms' => (int) $res->execution_time_ms,
                    'retrieved_sources' => json_decode($res->retrieved_sources ?? '{}', true)
                        ?: ['videos' => [], 'documents' => []],
                    'notes'             => $res->notes,
                ];
            }

            $export['test_runs'][] = [
                'run_id'           => (int) $run->id,
                'run_uuid'         => $run->run_uuid,
                'run_timestamp'    => gmdate('Y-m-d\TH:i:s\Z', (int) $run->created_time),
                'total_questions'  => (int) $run->question_count,
                'flagged_questions'=> (int) $run->flagged_count,
                'questions'        => $questions,
            ];
        }

        echo json_encode($export, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
        break;

    default:
        http_response_code(400);
        echo json_encode(['error' => 'Unknown action: ' . $action]);
        break;
}
