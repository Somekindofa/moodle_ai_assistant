<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * CLI re-ingest of course modules into ChromaDB.
 *
 * The web version (local/craftpilot/reingest_all.php) sets
 * set_time_limit(300). The corpus is ~571 supported modules / ~11,600 chunks,
 * ~8,970 of which need an LLM translation call, so a full run takes hours and
 * the web request dies long before it finishes. This script has no time limit
 * and is resumable, so an interrupted run costs nothing but the modules it had
 * not reached yet.
 *
 * Differences from the web version, all deliberate:
 *   - No time limit, no SSE, no sesskey (CLI has no session).
 *   - Does NOT clear local_craftpilot_cm_index unless --fresh is given. The
 *     web version clears it unconditionally at step 1, so an interrupted web
 *     run loses the record of everything that WAS indexed.
 *   - --resume skips modules already recorded in that table, which is what
 *     makes an interrupted run cheap to continue.
 *   - --course lets you pilot one course before committing to the whole
 *     corpus.
 *
 * Usage:
 *   # 1. Pilot on a single course and check the result first.
 *   php /var/www/html/public/local/craftpilot/cli/reingest_all.php --course=109
 *
 *   # 2. See what a full run would do, without doing it.
 *   php /var/www/html/public/local/craftpilot/cli/reingest_all.php --all --dry-run
 *
 *   # 3. The real thing. Use screen/tmux/nohup — it takes hours.
 *   php /var/www/html/public/local/craftpilot/cli/reingest_all.php --all
 *
 *   # 4. If it was interrupted, continue where it stopped.
 *   php /var/www/html/public/local/craftpilot/cli/reingest_all.php --all --resume
 *
 * BACK UP /opt/craftpilot_backend/chroma_langchain_db FIRST. This deletes and
 * rewrites each module's chunks as it goes.
 *
 * @package   local_craftpilot
 */

define('CLI_SCRIPT', true);
require(__DIR__ . '/../../../../config.php');
require_once($CFG->dirroot . '/lib/clilib.php');

// CLI scripts run without a session; capability checks inside the extractor
// need a user. Same call the migrate script uses.
\core\cron::setup_user();

// The whole point of this script: no wall clock, and room for big pages.
core_php_time_limit::raise(0);
raise_memory_limit(MEMORY_HUGE);

list($options, $unrecognised) = cli_get_params([
    'course'  => false,
    'all'     => false,
    'resume'  => false,
    'fresh'   => false,
    'dry-run' => false,
    'limit'   => 0,
    'wait'    => 180,
    'help'    => false,
], ['c' => 'course', 'a' => 'all', 'r' => 'resume', 'n' => 'dry-run', 'h' => 'help']);

if ($unrecognised) {
    cli_error('Unrecognised option(s): ' . implode(', ', $unrecognised));
}

if ($options['help'] || (!$options['all'] && $options['course'] === false)) {
    cli_writeln("
Re-ingest course modules into ChromaDB (no time limit, resumable).

Options:
  -c, --course=ID   Only this course. Use it to pilot before a full run.
  -a, --all         Every course. Takes hours - run under screen/tmux/nohup.
  -r, --resume      Skip modules already recorded as indexed. Use after an
                    interrupted run.
      --fresh       Clear local_craftpilot_cm_index first (what the web page
                    always does). Not needed for a normal re-ingest.
  -n, --dry-run     List what would be processed and exit. No changes, no
                    LLM calls, no cost.
      --limit=N     Stop after N modules. Handy with --dry-run.
      --wait=SECS   How long to wait for the backend to become ready
                    before giving up. Default 180. It takes ~1 minute to
                    start, and systemctl returns long before that.
  -h, --help        This message.

BACK UP /opt/craftpilot_backend/chroma_langchain_db BEFORE A REAL RUN.
");
    exit(0);
}

$dryrun   = (bool) $options['dry-run'];
$courseid = $options['course'] !== false ? (int) $options['course'] : 0;
$limit    = (int) $options['limit'];

if ($courseid && !$DB->record_exists('course', ['id' => $courseid])) {
    cli_error("Course {$courseid} does not exist.");
}

// ── Optionally clear the index table (the web page does this unconditionally)
if ($options['fresh'] && !$dryrun) {
    if ($courseid) {
        $DB->delete_records('local_craftpilot_cm_index', ['course_id' => $courseid]);
        cli_writeln("Cleared index records for course {$courseid}.");
    } else {
        $DB->delete_records('local_craftpilot_cm_index');
        cli_writeln('Cleared the whole index table.');
    }
}

// ── Collect modules ───────────────────────────────────────────────────────────
$supported = ['page', 'label', 'resource'];
$params    = $supported;
$where     = 'm.name IN (' . implode(',', array_fill(0, count($supported), '?')) . ')
              AND cm.deletioninprogress = 0';

if ($courseid) {
    $where   .= ' AND cm.course = ?';
    $params[] = $courseid;
}

$modules = $DB->get_records_sql("
    SELECT cm.id AS cmid,
           cm.course AS course_id,
           m.name AS modname,
           c.fullname AS course_name
      FROM {course_modules} cm
      JOIN {modules} m ON m.id = cm.module
      JOIN {course} c  ON c.id = cm.course
     WHERE {$where}
     ORDER BY cm.course, cm.id
", $params);

// --resume: drop anything already recorded as indexed.
if ($options['resume']) {
    $already = $DB->get_records_menu('local_craftpilot_cm_index', null, '', 'cmid, id');
    $before  = count($modules);
    $modules = array_filter($modules, function ($m) use ($already) {
        return !isset($already[(int) $m->cmid]);
    });
    cli_writeln('Resume: skipping ' . ($before - count($modules)) . ' module(s) already indexed.');
}

if ($limit > 0) {
    $modules = array_slice($modules, 0, $limit, true);
}

$total = count($modules);
if ($total === 0) {
    cli_writeln('Nothing to do.');
    exit(0);
}

cli_writeln(str_repeat('=', 68));
cli_writeln('CraftPilot re-ingest' . ($dryrun ? '  [DRY RUN - no changes]' : ''));
cli_writeln(($courseid ? "Scope    : course {$courseid}" : 'Scope    : ALL courses'));
cli_writeln("Modules  : {$total}");
cli_writeln(str_repeat('=', 68));

if ($dryrun) {
    foreach ($modules as $mod) {
        cli_writeln("  would process cmid={$mod->cmid} ({$mod->modname}) in \"{$mod->course_name}\"");
    }
    cli_writeln("\nDry run only. {$total} module(s) would be processed. Nothing changed.");
    exit(0);
}

// ── Preflight: wait for the backend to actually accept connections ─────
// The service is Type=simple, so `systemctl start` returns as soon as the
// process forks - roughly a minute before uvicorn binds the port. It loads
// embeddings, Chroma and the LLM client, then re-syncs annotations against a
// remote API, all before it listens. Without this check the script would march
// through every module turning connection refusals into errors, which is
// exactly what happened on the first pilot run. /api/health needs no token.
$healthurl = 'http://127.0.0.1:8000/api/health';
$deadline  = time() + max(0, (int) $options['wait']);
$ready     = false;
$announced = false;

do {
    $ch = curl_init($healthurl);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT        => 5,
        CURLOPT_FAILONERROR    => false,
    ]);
    curl_exec($ch);
    $code = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($code === 200) {
        $ready = true;
        break;
    }

    if (!$announced) {
        cli_writeln('Waiting for the CraftPilot backend to become ready (it takes ~1 minute to start)...');
        $announced = true;
    }
    sleep(3);
} while (time() < $deadline);

if (!$ready) {
    cli_error(
        "CraftPilot backend is not answering on {$healthurl}.\n" .
        "Nothing was changed. Check it with:\n" .
        "  systemctl status craftpilot-backend\n" .
        "  tail -30 /tmp/craftpilot_backend.log\n" .
        "Then re-run this script. Increase --wait if the machine is slow."
    );
}

cli_writeln('Backend is ready.');
cli_writeln('');

// ── Process ───────────────────────────────────────────────────────────────────
$extractor = new \local_craftpilot\course_content_extractor();
$client     = new \local_craftpilot\backend_client();

$done = $skipped = $errors = 0;
$i     = 0;
$start = microtime(true);

foreach ($modules as $mod) {
    $i++;
    $cmid     = (int) $mod->cmid;
    $courseId = (int) $mod->course_id;
    $modname  = $mod->modname;

    $elapsed = microtime(true) - $start;
    $eta     = $i > 1 ? sprintf(' eta %s', format_time((int) (($elapsed / ($i - 1)) * ($total - $i + 1)))) : '';
    $prefix  = sprintf('[%d/%d]%s', $i, $total, $eta);

    try {
        $payload = $extractor->extract_module($cmid, $modname, $courseId);
    } catch (\Throwable $e) {
        $errors++;
        cli_writeln("{$prefix} ERROR cmid={$cmid} ({$modname}): extract failed - " . $e->getMessage());
        continue;
    }

    if (empty($payload)) {
        $skipped++;
        cli_writeln("{$prefix} skip  cmid={$cmid} ({$modname}) - no extractable content");
        continue;
    }

    try {
        $client->delete_module($courseId, $cmid);
    } catch (\Throwable $e) {
        // Non-fatal, same as the web version: the ingest below rewrites it.
        cli_writeln("{$prefix} warn  cmid={$cmid}: delete failed - " . $e->getMessage());
    }

    try {
        $client->ingest_module($courseId, $cmid, $modname, $payload);
    } catch (\Throwable $e) {
        $errors++;
        cli_writeln("{$prefix} ERROR cmid={$cmid} ({$modname}): ingest failed - " . $e->getMessage());
        continue;
    }

    // Record it, so --resume can skip it if this run is interrupted.
    $md5      = isset($payload['content_html']) ? md5($payload['content_html']) : md5(json_encode($payload));
    $now      = time();
    $existing = $DB->get_record('local_craftpilot_cm_index', ['cmid' => $cmid]);

    if ($existing) {
        $DB->update_record('local_craftpilot_cm_index', (object) [
            'id'           => $existing->id,
            'content_hash' => $md5,
            'last_indexed' => $now,
        ]);
    } else {
        $DB->insert_record('local_craftpilot_cm_index', (object) [
            'cmid'         => $cmid,
            'course_id'    => $courseId,
            'content_hash' => $md5,
            'last_indexed' => $now,
        ]);
    }

    $done++;
    cli_writeln("{$prefix} ok    cmid={$cmid} ({$modname}) in \"{$mod->course_name}\"");
}

$took = format_time((int) (microtime(true) - $start));
cli_writeln(str_repeat('=', 68));
cli_writeln("Done in {$took}.  indexed={$done}  skipped={$skipped}  errors={$errors}  of {$total}");
if ($errors > 0) {
    cli_writeln('Some modules failed. Re-run with --resume to retry only those.');
}
cli_writeln(str_repeat('=', 68));
exit($errors > 0 ? 1 : 0);
