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
 * External API for managing conversation messages.
 *
 * @package   local_craftpilot
 */
class manage_messages extends external_api {

    public static function manage_messages_parameters() {
        return new external_function_parameters([
            'action'          => new external_value(PARAM_TEXT, 'Action to perform (save, load)'),
            'conversation_id' => new external_value(PARAM_TEXT, 'Conversation ID'),
            'message_type'    => new external_value(PARAM_TEXT, 'Message type (user or ai)', VALUE_DEFAULT, ''),
            'content'         => new external_value(PARAM_CLEANHTML, 'Message content', VALUE_DEFAULT, ''),
            'metadata'        => new external_value(PARAM_TEXT, 'Message metadata (JSON)', VALUE_DEFAULT, ''),
        ]);
    }

    public static function manage_messages(
        $action,
        $conversation_id,
        $message_type = '',
        $content = '',
        $metadata = ''
    ) {
        global $USER, $DB;

        $params = self::validate_parameters(self::manage_messages_parameters(), [
            'action'          => $action,
            'conversation_id' => $conversation_id,
            'message_type'    => $message_type,
            'content'         => $content,
            'metadata'        => $metadata,
        ]);

        $context = context_system::instance();
        self::validate_context($context);
        require_login();

        try {
            if (!$DB->get_manager()->table_exists('local_craftpilot_msg')) {
                return self::error_response('Database table local_craftpilot_msg does not exist');
            }
            if (!$DB->get_manager()->table_exists('local_craftpilot_conv')) {
                return self::error_response('Database table local_craftpilot_conv does not exist');
            }

            if (!self::user_owns_conversation($params['conversation_id'], $USER->id)) {
                return self::error_response('Access denied: You do not own this conversation');
            }

            switch ($params['action']) {
                case 'save':
                    return self::save_message(
                        $params['conversation_id'],
                        $params['message_type'],
                        $params['content'],
                        $params['metadata']
                    );
                case 'load':
                    return self::load_messages($params['conversation_id']);
                default:
                    return self::error_response('Invalid action specified');
            }

        } catch (\Exception $e) {
            error_log("CraftPilot manage_messages error: " . $e->getMessage());
            return self::error_response('Failed to manage messages: ' . $e->getMessage());
        }
    }

    private static function save_message(
        string $conversation_id,
        string $message_type,
        string $content,
        string $metadata = ''
    ): array {
        global $DB;

        if (empty($conversation_id)) {
            return self::error_response('Conversation ID is required');
        }
        if (empty($message_type) || !in_array($message_type, ['user', 'ai'])) {
            return self::error_response('Valid message type (user or ai) is required');
        }
        if (empty($content)) {
            return self::error_response('Message content is required');
        }

        $next_sequence = $DB->get_field_sql(
            'SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM {local_craftpilot_msg} WHERE conversation_id = ?',
            [$conversation_id]
        );

        $record                  = new \stdClass();
        $record->conversation_id = $conversation_id;
        $record->message_type    = $message_type;
        $record->content         = $content;
        $record->created_time    = time();
        $record->sequence_number = $next_sequence;
        $record->metadata        = $metadata;

        $id = $DB->insert_record('local_craftpilot_msg', $record);

        // Update conversation's last_updated timestamp.
        $conv = $DB->get_record('local_craftpilot_conv', ['conversation_id' => $conversation_id]);
        if ($conv) {
            $conv->last_updated = time();
            $DB->update_record('local_craftpilot_conv', $conv);
        }

        return [
            'success'         => true,
            'message'         => 'Message saved successfully',
            'message_id'      => $id,
            'sequence_number' => $next_sequence,
            'messages'        => [],
        ];
    }

    private static function load_messages(string $conversation_id): array {
        global $DB;

        if (empty($conversation_id)) {
            return self::error_response('Conversation ID is required');
        }

        $messages = $DB->get_records(
            'local_craftpilot_msg',
            ['conversation_id' => $conversation_id],
            'sequence_number ASC'
        );

        $list = [];
        foreach ($messages as $msg) {
            $list[] = [
                'id'              => $msg->id,
                'message_type'    => $msg->message_type,
                'content'         => $msg->content,
                'created_time'    => $msg->created_time,
                'sequence_number' => $msg->sequence_number,
                'metadata'        => $msg->metadata ?? '',
            ];
        }

        return [
            'success'  => true,
            'message'  => 'Messages loaded successfully',
            'messages' => $list,
        ];
    }

    private static function user_owns_conversation(string $conversation_id, int $user_id): bool {
        global $DB;
        $conv = $DB->get_record('local_craftpilot_conv', [
            'conversation_id' => $conversation_id,
            'userid'          => $user_id,
            'is_active'       => 1,
        ]);
        return $conv !== false;
    }

    private static function error_response(string $message): array {
        return [
            'success'  => false,
            'message'  => $message,
            'messages' => [],
        ];
    }

    public static function manage_messages_returns() {
        return new external_single_structure([
            'success'         => new external_value(PARAM_BOOL, 'Whether the operation was successful'),
            'message'         => new external_value(PARAM_TEXT, 'Status message'),
            'message_id'      => new external_value(PARAM_INT, 'Database message ID', VALUE_OPTIONAL),
            'sequence_number' => new external_value(PARAM_INT, 'Message sequence number', VALUE_OPTIONAL),
            'messages'        => new external_multiple_structure(
                new external_single_structure([
                    'id'              => new external_value(PARAM_INT, 'Database message ID'),
                    'message_type'    => new external_value(PARAM_TEXT, 'Message type (user or ai)'),
                    'content'         => new external_value(PARAM_RAW, 'Message content'),
                    'created_time'    => new external_value(PARAM_INT, 'Creation timestamp'),
                    'sequence_number' => new external_value(PARAM_INT, 'Message sequence number'),
                    'metadata'        => new external_value(PARAM_TEXT, 'Message metadata'),
                ]),
                'List of messages',
                VALUE_OPTIONAL
            ),
        ]);
    }
}
