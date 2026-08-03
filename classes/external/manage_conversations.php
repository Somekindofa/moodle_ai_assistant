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
use external_multiple_structure;
use context_system;

/**
 * External API for managing conversations (site-wide, no instance binding).
 *
 * @package   local_craftpilot
 */
class manage_conversations extends external_api {

    public static function manage_conversations_parameters() {
        return new external_function_parameters([
            'action'          => new external_value(PARAM_TEXT, 'Action to perform (create, list, update, delete)'),
            'conversation_id' => new external_value(PARAM_TEXT, 'Conversation ID', VALUE_DEFAULT, ''),
            'title'           => new external_value(PARAM_TEXT, 'Conversation title', VALUE_DEFAULT, ''),
            'metadata'        => new external_value(PARAM_TEXT, 'Conversation metadata (JSON)', VALUE_DEFAULT, ''),
            'courseid'        => new external_value(PARAM_INT, 'Course ID (retrieval hint)', VALUE_DEFAULT, 0),
        ]);
    }

    public static function manage_conversations(
        $action,
        $conversation_id = '',
        $title = '',
        $metadata = '',
        $courseid = 0
    ) {
        global $USER, $DB;

        $params = self::validate_parameters(self::manage_conversations_parameters(), [
            'action'          => $action,
            'conversation_id' => $conversation_id,
            'title'           => $title,
            'metadata'        => $metadata,
            'courseid'        => $courseid,
        ]);

        $context = context_system::instance();
        self::validate_context($context);
        require_login();

        try {
            if (!$DB->get_manager()->table_exists('local_craftpilot_conv')) {
                return self::error_response('Database table local_craftpilot_conv does not exist');
            }

            switch ($params['action']) {
                case 'create':
                    return self::create_conversation(
                        $USER->id,
                        $params['conversation_id'],
                        $params['title'],
                        $params['metadata'],
                        $params['courseid']
                    );
                case 'list':
                    return self::list_conversations($USER->id, $params['courseid']);
                case 'update':
                    return self::update_conversation(
                        $USER->id,
                        $params['conversation_id'],
                        $params['title'],
                        $params['metadata']
                    );
                case 'delete':
                    return self::delete_conversation($USER->id, $params['conversation_id']);
                default:
                    return self::error_response('Invalid action specified');
            }

        } catch (\Exception $e) {
            error_log("CraftPilot manage_conversations error: " . $e->getMessage());
            return self::error_response('Failed to manage conversation: ' . $e->getMessage());
        }
    }

    private static function create_conversation(
        int $user_id,
        string $conversation_id,
        string $title,
        string $metadata = '',
        int $courseid = 0
    ): array {
        global $DB;

        if (empty($conversation_id)) {
            return self::error_response('Conversation ID is required');
        }
        if (empty($title)) {
            return self::error_response('Conversation title is required');
        }

        $existing = $DB->get_record('local_craftpilot_conv', [
            'conversation_id' => $conversation_id,
            'is_active'       => 1,
        ]);
        if ($existing) {
            return self::error_response('Conversation with this ID already exists');
        }

        $record                  = new \stdClass();
        $record->conversation_id = $conversation_id;
        $record->userid          = $user_id;
        $record->courseid        = $courseid;
        $record->title           = $title;
        $record->created_time    = time();
        $record->last_updated    = time();
        $record->is_active       = 1;
        $record->metadata        = $metadata;

        $id = $DB->insert_record('local_craftpilot_conv', $record);

        return [
            'success'         => true,
            'message'         => 'Conversation created successfully',
            'conversation_id' => $conversation_id,
            'database_id'     => $id,
            'conversations'   => [],
        ];
    }

    private static function list_conversations(int $user_id, int $courseid = 0): array {
        global $DB;

        $conditions = ['userid' => $user_id, 'is_active' => 1];

        $conversations = $DB->get_records(
            'local_craftpilot_conv',
            $conditions,
            'last_updated DESC'
        );

        $list = [];
        foreach ($conversations as $conv) {
            $list[] = [
                'id'              => $conv->id,
                'conversation_id' => $conv->conversation_id,
                'title'           => $conv->title,
                'created_time'    => $conv->created_time,
                'last_updated'    => $conv->last_updated,
                'metadata'        => $conv->metadata ?? '',
            ];
        }

        return [
            'success'       => true,
            'message'       => 'Conversations retrieved successfully',
            'conversations' => $list,
        ];
    }

    private static function update_conversation(
        int $user_id,
        string $conversation_id,
        string $title,
        string $metadata = ''
    ): array {
        global $DB;

        if (empty($conversation_id)) {
            return self::error_response('Conversation ID is required');
        }

        $conv = $DB->get_record('local_craftpilot_conv', [
            'conversation_id' => $conversation_id,
            'userid'          => $user_id,
            'is_active'       => 1,
        ]);

        if (!$conv) {
            return self::error_response('Conversation not found or access denied');
        }

        if (!empty($title)) {
            $conv->title = $title;
        }
        if (!empty($metadata)) {
            $conv->metadata = $metadata;
        }
        $conv->last_updated = time();
        $DB->update_record('local_craftpilot_conv', $conv);

        return [
            'success'         => true,
            'message'         => 'Conversation updated successfully',
            'conversation_id' => $conversation_id,
            'conversations'   => [],
        ];
    }

    private static function delete_conversation(int $user_id, string $conversation_id): array {
        global $DB;

        if (empty($conversation_id)) {
            return self::error_response('Conversation ID is required');
        }

        $conv = $DB->get_record('local_craftpilot_conv', [
            'conversation_id' => $conversation_id,
            'userid'          => $user_id,
            'is_active'       => 1,
        ]);

        if (!$conv) {
            return self::error_response('Conversation not found or access denied');
        }

        $DB->delete_records('local_craftpilot_msg', ['conversation_id' => $conversation_id]);

        $conv->is_active    = 0;
        $conv->last_updated = time();
        $DB->update_record('local_craftpilot_conv', $conv);

        return [
            'success'         => true,
            'message'         => 'Conversation deleted successfully',
            'conversation_id' => $conversation_id,
            'conversations'   => [],
        ];
    }

    private static function error_response(string $message): array {
        return [
            'success'       => false,
            'message'       => $message,
            'conversations' => [],
        ];
    }

    public static function manage_conversations_returns() {
        return new external_single_structure([
            'success'         => new external_value(PARAM_BOOL, 'Whether the operation was successful'),
            'message'         => new external_value(PARAM_TEXT, 'Status message'),
            'conversation_id' => new external_value(PARAM_TEXT, 'Conversation ID', VALUE_OPTIONAL),
            'database_id'     => new external_value(PARAM_INT, 'Database record ID', VALUE_OPTIONAL),
            'conversations'   => new external_multiple_structure(
                new external_single_structure([
                    'id'              => new external_value(PARAM_INT, 'Database record ID'),
                    'conversation_id' => new external_value(PARAM_TEXT, 'Conversation ID'),
                    'title'           => new external_value(PARAM_TEXT, 'Conversation title'),
                    'created_time'    => new external_value(PARAM_INT, 'Creation timestamp'),
                    'last_updated'    => new external_value(PARAM_INT, 'Last update timestamp'),
                    'metadata'        => new external_value(PARAM_TEXT, 'Conversation metadata'),
                ]),
                'List of conversations',
                VALUE_OPTIONAL
            ),
        ]);
    }
}
