<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * AJAX endpoint: returns new bytes from /tmp/craftpilot_backend.log since the given offset.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

define('AJAX_SCRIPT', true);

require('../../config.php');

require_login();
require_capability('moodle/site:config', context_system::instance());

$offset  = optional_param('offset', -1, PARAM_INT);
$logfile = '/tmp/craftpilot_backend.log';

// Ensure offset cannot be a user-supplied arbitrary negative value beyond the -1 sentinel.
if ($offset < -1) {
    $offset = -1;
}

header('Content-Type: application/json');

if (!file_exists($logfile)) {
    echo json_encode(['lines' => [], 'offset' => 0]);
    exit;
}

clearstatcache(true, $logfile);
$size = filesize($logfile);

// First call (offset = -1): start from the last 4 KB to avoid dumping huge history.
if ($offset < 0) {
    $offset = max(0, $size - 4096);
}

// Clamp to valid file bounds.
$offset = min($offset, $size);

if ($offset >= $size) {
    echo json_encode(['lines' => [], 'offset' => $size]);
    exit;
}

$fh  = fopen($logfile, 'r');
fseek($fh, $offset);
$new = stream_get_contents($fh);
fclose($fh);

$lines = ($new !== false) ? explode("\n", $new) : [];

echo json_encode(['lines' => $lines, 'offset' => $size]);
