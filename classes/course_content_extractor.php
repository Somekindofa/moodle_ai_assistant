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
 * Extracts text content from supported Moodle course modules.
 *
 * Supported types:
 *   - mod_page   → HTML from page.content
 *   - mod_label  → HTML from label.intro
 *   - mod_resource (PDF/DOCX only) → raw bytes via Moodle file API
 *
 * @package   local_craftpilot
 */
class course_content_extractor {

    /** Maximum file size (bytes) we will base64-encode and send to the backend. */
    private const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10 MB

    /**
     * Extract content from any supported module type.
     *
     * @param int    $cmid      Course module ID
     * @param string $modname   Module type: 'page', 'label', or 'resource'
     * @param int    $course_id Moodle course ID
     * @return array Payload array or empty array on failure
     */
    public function extract_module(int $cmid, string $modname, int $course_id): array {
        switch ($modname) {
            case 'page':
                return $this->extract_page($cmid, $course_id);
            case 'label':
                return $this->extract_label($cmid, $course_id);
            case 'resource':
                return $this->extract_resource($cmid, $course_id);
            default:
                return [];
        }
    }

    // ─────────────────────────────────────────────────────────────
    // mod_page
    // ─────────────────────────────────────────────────────────────

    private function extract_page(int $cmid, int $course_id): array {
        global $DB;

        $cm = get_coursemodule_from_id('page', $cmid, $course_id, false, IGNORE_MISSING);
        if (!$cm) {
            error_log("CraftPilot extractor: page cmid={$cmid} not found");
            return [];
        }

        $page = $DB->get_record('page', ['id' => $cm->instance], 'id,name,content,contentformat', IGNORE_MISSING);
        if (!$page) {
            return [];
        }

        $section      = $DB->get_record('course_sections', ['id' => $cm->section], 'name,section', IGNORE_MISSING);
        $section_name = ($section && $section->name) ? $section->name : 'Section ' . ($section ? $section->section : '?');
        $context      = \context_module::instance($cmid);
        $html         = format_text($page->content, $page->contentformat, ['context' => $context, 'noclean' => true]);

        if (!trim(strip_tags($html))) {
            return [];
        }

        return [
            'course_id'    => (string) $course_id,
            'module_id'    => (string) $cmid,
            'module_type'  => 'page',
            'module_name'  => $page->name,
            'section_name' => $section_name,
            'content_html' => $html,
        ];
    }

    // ─────────────────────────────────────────────────────────────
    // mod_label
    // ─────────────────────────────────────────────────────────────

    private function extract_label(int $cmid, int $course_id): array {
        global $DB;

        $cm = get_coursemodule_from_id('label', $cmid, $course_id, false, IGNORE_MISSING);
        if (!$cm) {
            error_log("CraftPilot extractor: label cmid={$cmid} not found");
            return [];
        }

        $label = $DB->get_record('label', ['id' => $cm->instance], 'id,name,intro,introformat', IGNORE_MISSING);
        if (!$label) {
            return [];
        }

        $section      = $DB->get_record('course_sections', ['id' => $cm->section], 'name,section', IGNORE_MISSING);
        $section_name = ($section && $section->name) ? $section->name : 'Section ' . ($section ? $section->section : '?');
        $context      = \context_module::instance($cmid);
        $html         = format_text($label->intro, $label->introformat, ['context' => $context, 'noclean' => true]);

        if (!trim(strip_tags($html))) {
            return [];
        }

        return [
            'course_id'    => (string) $course_id,
            'module_id'    => (string) $cmid,
            'module_type'  => 'label',
            'module_name'  => $label->name ?: 'Label',
            'section_name' => $section_name,
            'content_html' => $html,
        ];
    }

    // ─────────────────────────────────────────────────────────────
    // mod_resource  (PDF / DOCX only)
    // ─────────────────────────────────────────────────────────────

    private function extract_resource(int $cmid, int $course_id): array {
        global $DB;

        $cm = get_coursemodule_from_id('resource', $cmid, $course_id, false, IGNORE_MISSING);
        if (!$cm) {
            error_log("CraftPilot extractor: resource cmid={$cmid} not found");
            return [];
        }

        $resource = $DB->get_record('resource', ['id' => $cm->instance], 'id,name', IGNORE_MISSING);
        if (!$resource) {
            return [];
        }

        $section      = $DB->get_record('course_sections', ['id' => $cm->section], 'name,section', IGNORE_MISSING);
        $section_name = ($section && $section->name) ? $section->name : 'Section ' . ($section ? $section->section : '?');
        $context      = \context_module::instance($cmid);

        $fs    = get_file_storage();
        $files = $fs->get_area_files($context->id, 'mod_resource', 'content', 0, 'sortorder DESC, id ASC', false);

        if (empty($files)) {
            error_log("CraftPilot extractor: no files for resource cmid={$cmid}");
            return [];
        }

        $file      = reset($files);
        $filename  = $file->get_filename();
        $extension = strtolower(pathinfo($filename, PATHINFO_EXTENSION));

        if (!in_array($extension, ['pdf', 'docx', 'doc'])) {
            return [];
        }

        $filesize = $file->get_filesize();
        if ($filesize > self::MAX_FILE_BYTES) {
            error_log("CraftPilot extractor: file {$filename} exceeds 10 MB limit — skipping");
            return [];
        }

        $raw_bytes   = $file->get_content();
        $content_b64 = base64_encode($raw_bytes);

        return [
            'course_id'       => (string) $course_id,
            'module_id'       => (string) $cmid,
            'module_type'     => 'resource',
            'module_name'     => $resource->name,
            'section_name'    => $section_name,
            'content_raw_b64' => $content_b64,
            'file_extension'  => $extension,
        ];
    }
}
