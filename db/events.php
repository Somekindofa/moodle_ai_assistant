<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Event observer registration for local_craftpilot.
 *
 * Keeps the ChromaDB vector store in sync with Moodle course content
 * automatically.  internal => false means the callback fires AFTER the
 * DB transaction commits — ensuring we read the final state.
 *
 * @package   local_craftpilot
 */

defined('MOODLE_INTERNAL') || die();

$observers = [
    [
        'eventname' => '\core\event\course_module_created',
        'callback'  => 'local_craftpilot\observer::course_module_created',
        'internal'  => false,
        'priority'  => 200,
    ],
    [
        'eventname' => '\core\event\course_module_updated',
        'callback'  => 'local_craftpilot\observer::course_module_updated',
        'internal'  => false,
        'priority'  => 200,
    ],
    [
        'eventname' => '\core\event\course_module_deleted',
        'callback'  => 'local_craftpilot\observer::course_module_deleted',
        'internal'  => false,
        'priority'  => 200,
    ],
    [
        'eventname' => '\core\event\course_deleted',
        'callback'  => 'local_craftpilot\observer::course_deleted',
        'internal'  => false,
        'priority'  => 200,
    ],
];
