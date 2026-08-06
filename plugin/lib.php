<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Library functions for the local_craftpilot plugin.
 *
 * local_craftpilot_before_footer() injects the widget HTML and queues the AMD
 * init on every authenticated page (fires just before </body>).
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

/**
 * Fires just before </body> — inject the widget DOM and queue the AMD init.
 *
 * Note: local_PLUGIN_page_init() is a theme-only callback and is never called
 * for local plugins. AMD initialisation must be queued here instead, while the
 * page requirements manager is still open (before $OUTPUT->footer() runs).
 *
 * @return string HTML to append before </body>
 */
function local_craftpilot_before_footer(): string {
    global $OUTPUT, $PAGE, $USER;

    if (!isloggedin() || isguestuser()) {
        return '';
    }

    // The video elicitation tool embeds its own full-bleed iframe via this Moodle
    // wrapper page. The pill would float on top of the iframe with nothing behind
    // it to chat about, so skip injection there.
    if (strpos($PAGE->url->get_path(), '/local/videoelicit/') === 0) {
        return '';
    }

    $courseid = isset($PAGE->course->id) ? (int) $PAGE->course->id : 0;
    $proxyurl = (new moodle_url('/local/craftpilot/chat_proxy.php'))->out(false);

    $PAGE->requires->js_call_amd('local_craftpilot/chat_interface', 'init', [
        $courseid,
        $proxyurl,
        (int) $USER->id,
    ]);

    return $OUTPUT->render_from_template('local_craftpilot/chat_interface', [
        'courseid' => $courseid,
    ]);
}
