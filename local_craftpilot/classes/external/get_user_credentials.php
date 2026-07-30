<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

namespace local_craftpilot\external;

defined('MOODLE_INTERNAL') || die();

require_once($CFG->libdir . '/externallib.php');

use external_api;
use external_function_parameters;
use external_value;
use external_single_structure;
use context_system;

/**
 * External API: check whether the backend is reachable (used by JS before streaming).
 *
 * @package   local_craftpilot
 */
class get_user_credentials extends external_api {

    public static function get_user_credentials_parameters() {
        return new external_function_parameters([]);
    }

    public static function get_user_credentials() {
        $context = context_system::instance();
        self::validate_context($context);
        require_login();

        return [
            'success'      => true,
            'api_key'      => '',
            'display_name' => '',
            'message'      => 'OK',
        ];
    }

    public static function get_user_credentials_returns() {
        return new external_single_structure([
            'success'      => new external_value(PARAM_BOOL,  'Whether the operation was successful'),
            'api_key'      => new external_value(PARAM_TEXT,  'Unused'),
            'display_name' => new external_value(PARAM_TEXT,  'Unused'),
            'message'      => new external_value(PARAM_TEXT,  'Status message'),
        ]);
    }
}
