<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Upgrade script for local_craftpilot.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

function xmldb_local_craftpilot_upgrade($oldversion) {
    global $DB;
    $dbman = $DB->get_manager();

    if ($oldversion < 2026031200) {
        // Drop the Fireworks API key table — no longer used.
        $table = new xmldb_table('local_craftpilot_keys');
        if ($dbman->table_exists($table)) {
            $dbman->drop_table($table);
        }

        upgrade_plugin_savepoint(true, 2026031200, 'local', 'craftpilot');
    }

    if ($oldversion < 2026032600) {
        // Create local_craftpilot_testrun table.
        $table = new xmldb_table('local_craftpilot_testrun');
        $table->add_field('id',             XMLDB_TYPE_INTEGER, '10',  null, XMLDB_NOTNULL, XMLDB_SEQUENCE);
        $table->add_field('run_uuid',       XMLDB_TYPE_CHAR,    '36',  null, XMLDB_NOTNULL);
        $table->add_field('created_time',   XMLDB_TYPE_INTEGER, '10',  null, XMLDB_NOTNULL);
        $table->add_field('question_count', XMLDB_TYPE_INTEGER, '5',   null, XMLDB_NOTNULL, null, '0');
        $table->add_field('flagged_count',  XMLDB_TYPE_INTEGER, '5',   null, XMLDB_NOTNULL, null, '0');
        $table->add_field('notes',          XMLDB_TYPE_TEXT,    null,  null);
        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_index('run_uuid_unique',  XMLDB_INDEX_UNIQUE,   ['run_uuid']);
        $table->add_index('created_time_idx', XMLDB_INDEX_NOTUNIQUE, ['created_time']);
        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        // Create local_craftpilot_testresult table.
        $table = new xmldb_table('local_craftpilot_testresult');
        $table->add_field('id',                XMLDB_TYPE_INTEGER, '10',  null, XMLDB_NOTNULL, XMLDB_SEQUENCE);
        $table->add_field('run_id',            XMLDB_TYPE_INTEGER, '10',  null, XMLDB_NOTNULL);
        $table->add_field('question_index',    XMLDB_TYPE_INTEGER, '5',   null, XMLDB_NOTNULL);
        $table->add_field('question_text',     XMLDB_TYPE_TEXT,    null,  null, XMLDB_NOTNULL);
        $table->add_field('generated_text',    XMLDB_TYPE_TEXT,    null,  null);
        $table->add_field('retrieved_sources', XMLDB_TYPE_TEXT,    null,  null);
        $table->add_field('refined_query',     XMLDB_TYPE_TEXT,    null,  null);
        $table->add_field('execution_time_ms', XMLDB_TYPE_INTEGER, '10',  null);
        $table->add_field('flagged',           XMLDB_TYPE_INTEGER, '1',   null, XMLDB_NOTNULL, null, '0');
        $table->add_field('notes',             XMLDB_TYPE_TEXT,    null,  null);
        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_key('run_fk',  XMLDB_KEY_FOREIGN, ['run_id'], 'local_craftpilot_testrun', ['id']);
        $table->add_index('run_question_unique', XMLDB_INDEX_UNIQUE,    ['run_id', 'question_index']);
        $table->add_index('flagged_idx',         XMLDB_INDEX_NOTUNIQUE, ['flagged']);
        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        upgrade_plugin_savepoint(true, 2026032600, 'local', 'craftpilot');
    }

    return true;
}
