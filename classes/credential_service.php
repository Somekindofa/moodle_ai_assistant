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
 * Service class for managing CraftPilot backend credentials.
 *
 * @package   local_craftpilot
 */
class credential_service {

    /**
     * Get the internal API token used to authenticate Moodle → backend requests.
     *
     * @return string|null
     */
    public static function get_internal_api_token(): ?string {
        $token = get_config('local_craftpilot', 'internal_api_token');
        return !empty($token) ? $token : null;
    }
}
