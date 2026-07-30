<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

namespace local_craftpilot;

defined('MOODLE_INTERNAL') || die();

/**
 * HTTP client for the CraftPilot Python backend.
 *
 * @package   local_craftpilot
 */
class backend_client {

    /** Backend base URL — same host as the chat proxy. */
    private const BASE_URL = 'http://127.0.0.1:8000/api';

    /** cURL timeout in seconds for ingestion calls (large files may take longer). */
    private const TIMEOUT = 60;

    // ─────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────

    public function ingest_module(int $course_id, int $cmid, string $modname, array $payload): void {
        $this->post('/ingest-course-module', $payload);
        error_log("CraftPilot backend_client: ingested course={$course_id} module={$cmid} type={$modname}");
    }

    public function delete_module(int $course_id, int $cmid): void {
        $this->delete('/delete-course-module', [
            'course_id' => (string) $course_id,
            'module_id' => (string) $cmid,
        ]);
        error_log("CraftPilot backend_client: deleted course={$course_id} module={$cmid}");
    }

    public function delete_course(int $course_id): void {
        $this->delete('/delete-course', ['course_id' => (string) $course_id]);
        error_log("CraftPilot backend_client: deleted course collection course={$course_id}");
    }

    /**
     * Run a single chat question against the RAG backend and collect the full response.
     *
     * Posts to /api/chat with CURLOPT_RETURNTRANSFER (non-streaming) and parses the
     * newline-delimited JSON event stream into a structured result array.
     *
     * @param  string      $message         User question text.
     * @param  string      $conversation_id Unique conversation thread ID (e.g. "{run_uuid}-q{n}").
     * @param  string|null $selected_domain Optional craft domain filter.
     * @param  string|null $course_id       Optional Moodle course ID filter.
     * @return array{generated_text: string, retrieved_sources: array, refined_query: string|null, execution_time_ms: int}
     * @throws \RuntimeException on cURL error or non-2xx HTTP status.
     */
    public function chat_request_full(
        string  $message,
        string  $conversation_id,
        ?string $selected_domain = null,
        ?string $course_id       = null
    ): array {
        $payload = [
            'message'                => $message,
            'conversation_thread_id' => $conversation_id,
            'is_first_message'       => true,
            'selected_domain'        => $selected_domain,
            'course_id'              => $course_id,
        ];

        $started = microtime(true);
        $raw     = $this->post_raw('/chat', $payload);
        $elapsed = (int) round((microtime(true) - $started) * 1000);

        $generated_text    = '';
        $retrieved_sources = ['videos' => [], 'documents' => []];
        $refined_query     = null;

        foreach (explode("\n", $raw) as $line) {
            $line = trim($line);
            if ($line === '' || $line === '[DONE]') {
                continue;
            }
            $obj = json_decode($line, true);
            if (!is_array($obj)) {
                continue;
            }

            $event = $obj['event'] ?? null;

            if ($event === 'token') {
                $generated_text .= $obj['data'] ?? '';
            } elseif ($event === 'video_metadata') {
                $meta = $obj['data'] ?? [];
                $retrieved_sources['videos'][] = [
                    'video_id'     => $meta['video_id']     ?? null,
                    'filepath'     => $meta['filepath']     ?? null,
                    'filename'     => $meta['filename']     ?? null,
                    'start_time'   => $meta['start_time']   ?? null,
                    'end_time'     => $meta['end_time']     ?? null,
                    'video_url'    => $meta['video_url']    ?? null,
                    'project_name' => $meta['project_name'] ?? null,
                ];
            } elseif ($event === 'documents') {
                foreach (($obj['data'] ?? []) as $doc) {
                    $retrieved_sources['documents'][] = [
                        'module_name'  => $doc['module_name']  ?? null,
                        'heading_path' => $doc['heading_path'] ?? null,
                        'content'      => substr($doc['content'] ?? '', 0, 400),
                        'course_id'    => $doc['course_id']    ?? null,
                    ];
                }
            } elseif ($event === 'refined_query') {
                $refined_query = $obj['data'] ?? null;
            }
            // conversation_title and terminal [DONE] marker are silently ignored.
        }

        return [
            'generated_text'    => trim($generated_text),
            'retrieved_sources' => $retrieved_sources,
            'refined_query'     => $refined_query,
            'execution_time_ms' => $elapsed,
        ];
    }

    // ─────────────────────────────────────────────────────────────
    // cURL helpers
    // ─────────────────────────────────────────────────────────────

    private function post(string $path, array $data): array {
        return $this->request('POST', $path, $data);
    }

    /**
     * POST to the backend and return the raw response body as a string.
     * Used for the streaming /chat endpoint which returns newline-delimited JSON,
     * not a single decodable JSON object.
     */
    private function post_raw(string $path, array $data): string {
        $url   = self::BASE_URL . $path;
        $body  = json_encode($data, JSON_UNESCAPED_UNICODE);
        $token = get_config('local_craftpilot', 'internal_api_token');

        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_POST           => true,
            CURLOPT_POSTFIELDS     => $body,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => 120,
            CURLOPT_CONNECTTIMEOUT => 10,
            CURLOPT_HTTPHEADER     => [
                'Content-Type: application/json',
                'Content-Length: ' . strlen($body),
                'X-Internal-Token: ' . $token,
            ],
        ]);

        $response   = curl_exec($ch);
        $http_code  = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $curl_error = curl_error($ch);
        curl_close($ch);

        if ($curl_error) {
            throw new \RuntimeException("CraftPilot backend cURL error [POST {$path}]: {$curl_error}");
        }

        if ($http_code >= 400) {
            throw new \RuntimeException(
                "CraftPilot backend HTTP {$http_code} [POST {$path}]: " . substr((string) $response, 0, 300)
            );
        }

        return (string) $response;
    }

    private function delete(string $path, array $data): array {
        return $this->request('DELETE', $path, $data);
    }

    private function request(string $method, string $path, array $data): array {
        $url   = self::BASE_URL . $path;
        $body  = json_encode($data, JSON_UNESCAPED_UNICODE);
        $token = get_config('local_craftpilot', 'internal_api_token');

        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_CUSTOMREQUEST  => $method,
            CURLOPT_POSTFIELDS     => $body,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => self::TIMEOUT,
            CURLOPT_HTTPHEADER     => [
                'Content-Type: application/json',
                'Content-Length: ' . strlen($body),
                'X-Internal-Token: ' . $token,
            ],
        ]);

        $response   = curl_exec($ch);
        $http_code  = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $curl_error = curl_error($ch);
        curl_close($ch);

        if ($curl_error) {
            throw new \RuntimeException("CraftPilot backend cURL error [{$method} {$path}]: {$curl_error}");
        }

        if ($http_code >= 400) {
            throw new \RuntimeException(
                "CraftPilot backend HTTP {$http_code} [{$method} {$path}]: " . substr($response, 0, 500)
            );
        }

        $decoded = json_decode($response, true);
        return is_array($decoded) ? $decoded : [];
    }
}
