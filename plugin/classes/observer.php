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
 * Event observer for local_craftpilot.
 *
 * Watches core Moodle course module events and keeps the Python backend's
 * ChromaDB vector store in sync with course content.
 *
 * Only mod_page, mod_label, and mod_resource (PDF/DOCX) are indexed.
 * All HTTP errors are caught and logged — never re-thrown — so that a
 * backend failure cannot affect the Moodle teacher's save operation.
 *
 * @package   local_craftpilot
 */
class observer {

    /** Module types we index. */
    private const SUPPORTED_TYPES = ['page', 'label', 'resource'];

    // ─────────────────────────────────────────────────────────────
    // Event handlers
    // ─────────────────────────────────────────────────────────────

    public static function course_module_created(\core\event\course_module_created $event): void {
        $modname = self::get_modname($event);
        if (!in_array($modname, self::SUPPORTED_TYPES)) {
            return;
        }

        $cmid      = (int) $event->objectid;
        $course_id = (int) $event->courseid;

        try {
            $extractor = new course_content_extractor();
            $payload   = $extractor->extract_module($cmid, $modname, $course_id);
            if (empty($payload)) {
                return;
            }

            $hash = self::content_hash($payload);
            self::upsert_index_record($cmid, $course_id, $hash);

            $client = new backend_client();
            $client->ingest_module($course_id, $cmid, $modname, $payload);

        } catch (\Throwable $e) {
            error_log("CraftPilot observer [created] cmid={$cmid}: " . $e->getMessage());
        }
    }

    public static function course_module_updated(\core\event\course_module_updated $event): void {
        $modname = self::get_modname($event);
        if (!in_array($modname, self::SUPPORTED_TYPES)) {
            return;
        }

        $cmid      = (int) $event->objectid;
        $course_id = (int) $event->courseid;

        try {
            $extractor = new course_content_extractor();
            $payload   = $extractor->extract_module($cmid, $modname, $course_id);
            if (empty($payload)) {
                return;
            }

            $hash = self::content_hash($payload);

            // Skip if content has not changed (saves an embedding round-trip).
            $existing = self::get_index_record($cmid);
            if ($existing && $existing->content_hash === $hash) {
                return;
            }

            $client = new backend_client();
            // Remove stale chunks first, then ingest fresh ones.
            $client->delete_module($course_id, $cmid);
            $client->ingest_module($course_id, $cmid, $modname, $payload);

            self::upsert_index_record($cmid, $course_id, $hash);

        } catch (\Throwable $e) {
            error_log("CraftPilot observer [updated] cmid={$cmid}: " . $e->getMessage());
        }
    }

    public static function course_module_deleted(\core\event\course_module_deleted $event): void {
        $cmid      = (int) $event->objectid;
        $course_id = (int) $event->courseid;

        try {
            $client = new backend_client();
            $client->delete_module($course_id, $cmid);
            self::delete_index_record($cmid);
        } catch (\Throwable $e) {
            error_log("CraftPilot observer [deleted] cmid={$cmid}: " . $e->getMessage());
        }
    }

    public static function course_deleted(\core\event\course_deleted $event): void {
        $course_id = (int) $event->objectid;

        try {
            $client = new backend_client();
            $client->delete_course($course_id);
            self::delete_course_index_records($course_id);
        } catch (\Throwable $e) {
            error_log("CraftPilot observer [course_deleted] course_id={$course_id}: " . $e->getMessage());
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────

    /**
     * Safely extract the module type name from an event.
     */
    private static function get_modname(\core\event\base $event): string {
        $other = $event->other ?? [];
        if (!empty($other['modulename'])) {
            return (string) $other['modulename'];
        }
        global $DB;
        $cmid = (int) $event->objectid;
        try {
            $cm = $DB->get_record_sql(
                'SELECT m.name FROM {course_modules} cm JOIN {modules} m ON m.id = cm.module WHERE cm.id = ?',
                [$cmid],
                IGNORE_MISSING
            );
            return $cm ? (string) $cm->name : '';
        } catch (\Throwable $e) {
            return '';
        }
    }

    /** MD5 of the text content extracted from a payload. */
    private static function content_hash(array $payload): string {
        $text = isset($payload['content_html']) ? $payload['content_html'] : '';
        if (!$text && isset($payload['content_raw_b64'])) {
            $text = $payload['content_raw_b64'];
        }
        return md5($text);
    }

    // ─────────────────────────────────────────────────────────────
    // local_craftpilot_cm_index DB helpers
    // ─────────────────────────────────────────────────────────────

    private static function get_index_record(int $cmid): ?\stdClass {
        global $DB;
        return $DB->get_record('local_craftpilot_cm_index', ['cmid' => $cmid]) ?: null;
    }

    private static function upsert_index_record(int $cmid, int $course_id, string $hash): void {
        global $DB;
        $now      = time();
        $existing = self::get_index_record($cmid);
        if ($existing) {
            $existing->content_hash = $hash;
            $existing->last_indexed = $now;
            $DB->update_record('local_craftpilot_cm_index', $existing);
        } else {
            $rec               = new \stdClass();
            $rec->cmid         = $cmid;
            $rec->course_id    = $course_id;
            $rec->content_hash = $hash;
            $rec->last_indexed = $now;
            $DB->insert_record('local_craftpilot_cm_index', $rec);
        }
    }

    private static function delete_index_record(int $cmid): void {
        global $DB;
        $DB->delete_records('local_craftpilot_cm_index', ['cmid' => $cmid]);
    }

    private static function delete_course_index_records(int $course_id): void {
        global $DB;
        $DB->delete_records('local_craftpilot_cm_index', ['course_id' => $course_id]);
    }
}
