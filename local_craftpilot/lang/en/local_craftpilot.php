<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Language strings for the local_craftpilot plugin.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'CraftPilot';

// Chat interface strings
$string['promptplaceholder'] = 'Ask a question about this content...';
$string['send'] = 'Send';
$string['openchat'] = 'Open CraftPilot chat';
$string['closechat'] = 'Close chat';
$string['retrieveddocs'] = 'Retrieved Documents';
$string['sources'] = 'Sources';
$string['newconversation'] = 'New conversation';
$string['conversations'] = 'Conversations';

// Admin panel strings
$string['adminpanel']     = 'CraftPilot Backend Admin';
$string['backendstatus']  = 'Backend Status';
$string['indexedcourses'] = 'Indexed Course Collections';
$string['reingestall']    = 'Re-ingest All Courses';
$string['reingestdesc']   = 'Rebuilds ChromaDB from all Moodle pages, labels and resource files. Use after a vector store wipe or import.';
$string['reingestbtn']    = 'Re-ingest All';
$string['livebackendlog'] = 'Live Backend Log';
$string['autoscroll']     = 'Auto-scroll';
$string['clearview']      = 'Clear view';
$string['vectordoccount'] = 'Vector docs';
$string['lastindexed']    = 'Last indexed';

// Backend Configuration
$string['backend_heading'] = 'Backend Configuration';
$string['backend_heading_desc'] = 'Configure the connection between the Moodle plugin and the Python backend.';
$string['internal_api_token'] = 'Internal API Token';
$string['internal_api_token_desc'] = 'Shared secret used to authenticate Moodle → backend requests. Must match INTERNAL_API_TOKEN in the backend .env file.';

// Test Bench
$string['testbench']            = 'CraftPilot RAG Test Bench';
$string['testbenchdesc']        = 'Run and inspect MOCO 2026 evaluation questions against the RAG pipeline.';
$string['runtests']             = 'Run Tests';
$string['exportflagged']        = 'Export Flagged';
$string['testrunhistory']       = 'Run History';
$string['notestresults']        = 'No test results yet. Click "Run Tests" to begin.';
$string['testbenchquestions']   = 'Test Questions';
$string['tbscenario']           = 'Scenario';
