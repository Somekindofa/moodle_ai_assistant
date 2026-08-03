<?php
/**
 * Migration CLI script: mod_craftpilot → local_craftpilot
 *
 * Reads all mod_craftpilot activity instances that have non-empty content and
 * creates equivalent mod_page course modules.  The existing
 * local_craftpilot event observer will fire course_module_created for each
 * new page and ingest it into ChromaDB automatically.
 *
 * Usage:
 *   php /var/www/html/public/local/craftpilot/cli/migrate_from_mod.php
 *
 * Safe to run multiple times — skips instances whose content is empty.
 * After verifying output, uninstall mod_craftpilot via the Moodle admin UI.
 *
 * @package   local_craftpilot
 */

define('CLI_SCRIPT', true);
require(__DIR__ . '/../../../../config.php');
require_once($CFG->dirroot . '/lib/clilib.php');
require_once($CFG->dirroot . '/course/modlib.php');

// CLI scripts run without a session; set up the site admin user so that
// capability checks inside create_module() pass.
\core\cron::setup_user();

// Verify mod_craftpilot table exists.
if (!$DB->get_manager()->table_exists('craftpilot')) {
    cli_error('Table mdl_craftpilot does not exist — is mod_craftpilot installed?');
}

// Verify local_craftpilot tables exist.
if (!$DB->get_manager()->table_exists('local_craftpilot_conv')) {
    cli_error('Table mdl_local_craftpilot_conv does not exist — run upgrade.php first.');
}

cli_writeln("CraftPilot Migration: mod_craftpilot → Moodle Pages");
cli_writeln(str_repeat('-', 60));

$instances = $DB->get_records('craftpilot', null, 'course ASC, id ASC');
$created   = 0;
$skipped   = 0;

foreach ($instances as $instance) {
    $content = trim(strip_tags($instance->content ?? ''));
    if (empty($content)) {
        cli_writeln("  SKIP  course={$instance->course} id={$instance->id} name=\"{$instance->name}\" (empty content)");
        $skipped++;
        continue;
    }

    // Find the section number of the original course module.
    $cm = $DB->get_record_sql(
        'SELECT cm.section FROM {course_modules} cm
         JOIN {modules} m ON m.id = cm.module
         WHERE m.name = ? AND cm.instance = ? AND cm.course = ?',
        ['craftpilot', $instance->id, $instance->course],
        IGNORE_MISSING
    );

    $section_num = 0;
    if ($cm) {
        $sec = $DB->get_record('course_sections', ['id' => $cm->section], 'section', IGNORE_MISSING);
        if ($sec) {
            $section_num = (int) $sec->section;
        }
    }

    try {
        // create_module() triggers the course_module_created event which
        // the local_craftpilot observer handles for ChromaDB ingestion.
        $module = [
            'modulename'     => 'page',
            'course'         => $instance->course,
            'name'           => '[CraftPilot] ' . $instance->name,
            'introeditor'    => ['text' => '', 'format' => FORMAT_HTML, 'itemid' => 0],
            'content'        => $instance->content,
            'contentformat'  => $instance->contentformat ?? FORMAT_HTML,
            'display'        => 0,
            'section'        => $section_num,
            'visible'        => 1,
        ];

        $cm_info = create_module((object) $module);
        cli_writeln("  CREATE course={$instance->course} id={$instance->id} name=\"{$instance->name}\" → page cmid={$cm_info->coursemodule}");
        $created++;

    } catch (Exception $e) {
        cli_writeln("  ERROR  course={$instance->course} id={$instance->id}: " . $e->getMessage());
    }
}

cli_writeln(str_repeat('-', 60));
cli_writeln("Done. Created: {$created}  Skipped (empty): {$skipped}");
cli_writeln("");
cli_writeln("Next steps:");
cli_writeln("  1. Verify pages were created: Site Admin → Course Management");
cli_writeln("  2. Check ChromaDB ingestion: tail -f /tmp/craftpilot_backend.log | grep Indexed");
cli_writeln("  3. Uninstall mod_craftpilot: Site Admin → Plugins → Activity Modules → CraftPilot → Uninstall");
cli_writeln("  4. Purge caches: php admin/cli/purge_caches.php");
