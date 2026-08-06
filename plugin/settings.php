<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Settings for the local_craftpilot plugin.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

if ($hassiteconfig) {
    $settings = new admin_settingpage('local_craftpilot', get_string('pluginname', 'local_craftpilot'));
    $ADMIN->add('localplugins', $settings);

    $settings->add(new admin_setting_heading(
        'local_craftpilot/backend_heading',
        get_string('backend_heading', 'local_craftpilot'),
        get_string('backend_heading_desc', 'local_craftpilot')
    ));

    $settings->add(new admin_setting_configpasswordunmask(
        'local_craftpilot/internal_api_token',
        get_string('internal_api_token', 'local_craftpilot'),
        get_string('internal_api_token_desc', 'local_craftpilot'),
        ''
    ));

    $ADMIN->add('localplugins', new admin_externalpage(
        'local_craftpilot_adminpanel',
        get_string('adminpanel', 'local_craftpilot'),
        new moodle_url('/local/craftpilot/admin_panel.php'),
        'moodle/site:config'
    ));

    $ADMIN->add('localplugins', new admin_externalpage(
        'local_craftpilot_testbench',
        get_string('testbench', 'local_craftpilot'),
        new moodle_url('/local/craftpilot/test_bench.php'),
        'moodle/site:config'
    ));
}
