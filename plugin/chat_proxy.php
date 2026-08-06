<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Session-validation proxy for the RAG backend streaming chat.
 *
 * Validates the Moodle session and sesskey, then issues an HTTP 307
 * redirect to /craftpilot-api/chat so the browser streams the response
 * directly through Apache's ProxyPass (with flushpackets=on), bypassing
 * PHP-FPM's buffered FastCGI transport entirely.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require('../../../config.php');

ignore_user_abort(true);
set_time_limit(0);

// Must be logged in.
require_login();

// Only accept POST requests.
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Method not allowed']);
    exit;
}

// Read and validate the incoming JSON body.
$rawbody = file_get_contents('php://input');
$data    = json_decode($rawbody, true);

if (!is_array($data) || empty($data['conversation_thread_id'])) {
    http_response_code(400);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Missing or invalid request body']);
    exit;
}

// Validate the sesskey included in the JSON body.
$incoming_sesskey = $data['sesskey'] ?? '';
if (!confirm_sesskey($incoming_sesskey)) {
    http_response_code(403);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Invalid session key']);
    exit;
}

// Validate that the user_id in the body matches the authenticated session.
// This prevents a logged-in user from claiming another user's identity.
$incoming_user_id = isset($data['user_id']) ? (int)$data['user_id'] : 0;
if ($incoming_user_id !== (int)$USER->id) {
    http_response_code(403);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'user_id mismatch']);
    exit;
}

// Session valid. Issue a 307 (method-preserving) redirect so the browser
// re-POSTs the same body to the Apache proxy endpoint. Apache injects the
// X-Internal-Token header automatically for /craftpilot-api/ requests, and
// the ProxyPass uses flushpackets=on so status events reach the browser
// immediately without FastCGI buffering.
header('Location: /craftpilot-api/chat', true, 307);
exit;
