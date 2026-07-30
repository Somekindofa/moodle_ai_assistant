<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Fixed evaluation questions for the MOCO 2026 RAG test bench.
 *
 * These questions are anchored for reproducible evaluation across runs.
 * They cover distinct retrieval scenarios: novice vocabulary, expert vocabulary,
 * video retrieval, course content retrieval, hybrid, and the off-topic guardrail.
 *
 * @package   local_craftpilot
 * @copyright 2026
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

/**
 * MOCO 2026 evaluation question set.
 * Each entry: id (string), label (string), scenario (string), text (string).
 */
const TESTBENCH_QUESTIONS = [
    [
        'id'       => 'Q01',
        'label'    => 'Vocabulaire novice — geste',
        'scenario' => 'novice_vocabulary',
        'text'     => "Je débute en maroquinerie. Comment faire pour couper le cuir proprement sans le déchirer ?",
    ],
    [
        'id'       => 'Q02',
        'label'    => 'Vocabulaire expert — technique',
        'scenario' => 'expert_vocabulary',
        'text'     => "Quelle est la différence entre un refente et un parage dans la préparation des peaux, et dans quel ordre les réalise-t-on ?",
    ],
    [
        'id'       => 'Q03',
        'label'    => 'Récupération vidéo — outil spécifique',
        'scenario' => 'video_retrieval',
        'text'     => "Comment utilise-t-on le couteau à parer pour amincir les bords d'une pièce de cuir ?",
    ],
    [
        'id'       => 'Q04',
        'label'    => 'Récupération vidéo — geste corporel',
        'scenario' => 'video_retrieval',
        'text'     => "Comment tenir correctement l'alène lors du piquage main pour éviter les blessures et avoir des points réguliers ?",
    ],
    [
        'id'       => 'Q05',
        'label'    => 'Récupération cours — sécurité',
        'scenario' => 'course_content_retrieval',
        'text'     => "Quelles sont les règles de sécurité à respecter lors de l'utilisation des machines de coupe en atelier ?",
    ],
    [
        'id'       => 'Q06',
        'label'    => 'Récupération cours — matériaux',
        'scenario' => 'course_content_retrieval',
        'text'     => "Quels types de cuir sont recommandés pour la réalisation d'un sac à main structuré, et pourquoi ?",
    ],
    [
        'id'       => 'Q07',
        'label'    => 'Hybride vidéo + cours — assemblage',
        'scenario' => 'hybrid_retrieval',
        'text'     => "Explique les étapes du montage d'une anse sur un sac : depuis la préparation des pièces jusqu'à la finition.",
    ],
    [
        'id'       => 'Q08',
        'label'    => 'Hors-sujet — garde-fou',
        'scenario' => 'guardrail',
        'text'     => "Peux-tu me donner une recette de cuisine facile à préparer pour le dîner ce soir ?",
    ],
];
