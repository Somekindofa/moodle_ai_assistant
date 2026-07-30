<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * External functions and services for the local_craftpilot plugin.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$functions = [
    'local_craftpilot_get_user_credentials' => [
        'classname'     => 'local_craftpilot\external\get_user_credentials',
        'methodname'    => 'get_user_credentials',
        'description'   => 'Check session validity before streaming',
        'type'          => 'read',
        'ajax'          => true,
        'loginrequired' => true,
    ],
    'local_craftpilot_manage_conversations' => [
        'classname'     => 'local_craftpilot\external\manage_conversations',
        'methodname'    => 'manage_conversations',
        'description'   => 'Manage site-wide conversations',
        'type'          => 'write',
        'ajax'          => true,
        'loginrequired' => true,
    ],
    'local_craftpilot_manage_messages' => [
        'classname'     => 'local_craftpilot\external\manage_messages',
        'methodname'    => 'manage_messages',
        'description'   => 'Manage conversation messages (save, load)',
        'type'          => 'write',
        'ajax'          => true,
        'loginrequired' => true,
    ],
];
