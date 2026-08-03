<?php
/**
 * Video streaming proxy for the RAG backend.
 *
 * The FastAPI backend runs on 127.0.0.1:8000 and is unreachable from the
 * browser.  This script:
 *   1. Validates the Moodle session.
 *   2. Forwards the GET request (including any Range header) to the backend.
 *   3. Relays the response headers and streams the binary video data back.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require('../../../config.php');
require_login();

// Only GET is meaningful for video streaming.
if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
    http_response_code(405);
    exit;
}

// Validate the video_id parameter — must be an MD5 hex string.
$video_id = isset($_GET['id']) ? trim($_GET['id']) : '';
if (!preg_match('/^[a-f0-9]{32}$/', $video_id)) {
    http_response_code(400);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Invalid video ID']);
    exit;
}

$backend_url    = 'http://127.0.0.1:8000/api/video/stream/' . $video_id;
$internal_token = get_config('local_craftpilot', 'internal_api_token');

$curl_headers = [
    'X-Internal-Token: ' . $internal_token,
];

// Forward the Range header if the browser sent one (required for seeking).
if (isset($_SERVER['HTTP_RANGE'])) {
    $curl_headers[] = 'Range: ' . $_SERVER['HTTP_RANGE'];
}

$ch = curl_init();
curl_setopt_array($ch, [
    CURLOPT_URL            => $backend_url,
    CURLOPT_HTTPGET        => true,
    CURLOPT_HTTPHEADER     => $curl_headers,
    CURLOPT_TIMEOUT        => 0,        // no timeout — video can be long
    CURLOPT_CONNECTTIMEOUT => 10,
    CURLOPT_RETURNTRANSFER => false,    // stream directly
    CURLOPT_FOLLOWLOCATION => false,

    // Relay the response status code and headers before streaming body.
    CURLOPT_HEADERFUNCTION => function ($ch, $header) {
        $len    = strlen($header);
        $header = trim($header);

        if ($header === '') {
            return $len;
        }

        // Set the HTTP response code from the status line.
        if (preg_match('/^HTTP\/[\d.]+ (\d{3})/', $header, $m)) {
            http_response_code((int)$m[1]);
            return $len;
        }

        // Relay whichever headers matter for video playback.
        foreach (['Content-Type', 'Content-Length', 'Content-Range', 'Accept-Ranges'] as $relay) {
            if (stripos($header, $relay . ':') === 0) {
                header($header, true);
                return $len;
            }
        }

        return $len;
    },

    // Stream body bytes directly to the browser.
    CURLOPT_WRITEFUNCTION  => function ($ch, $chunk) {
        echo $chunk;
        flush();
        return strlen($chunk);
    },
]);

// Disable any output buffering so bytes reach the browser immediately.
while (ob_get_level()) {
    ob_end_flush();
}
ob_implicit_flush(true);

$ok      = curl_exec($ch);
$curlerr = curl_error($ch);
curl_close($ch);

if (!$ok) {
    http_response_code(502);
}
