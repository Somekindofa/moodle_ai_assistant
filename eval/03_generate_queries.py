"""
Step 3: Generate synthetic queries using the Infomaniak LLM API.
Saves eval/fixtures/ground_truth.json.
"""

import os
import sys
import json
import time
import logging
import re

os.chdir('/opt/craftpilot_backend')
sys.path.insert(0, '/opt/craftpilot_backend')

logging.basicConfig(level=logging.WARNING)

from openai import OpenAI

FIXTURES_DIR = '/opt/craftpilot_backend/eval/fixtures'
os.makedirs(FIXTURES_DIR, exist_ok=True)

INFOMANIAK_API_KEY = "REDACTED_OLD_INFOMANIAK_KEY"
INFOMANIAK_PRODUCT_ID = "106980"
LLM_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"

client = OpenAI(
    api_key=INFOMANIAK_API_KEY,
    base_url=f"https://api.infomaniak.com/2/ai/{INFOMANIAK_PRODUCT_ID}/openai/v1"
)

# ── Annotation data ───────────────────────────────────────────────────────────

ANNOTATIONS = {
    17: {
        "video_filename": "GX010191.MP4",
        "start_time": 110.712,
        "end_time": 135.822,
        "duration": 25.11,
        "transcription": (
            "J'allume la flamme avec l'arc électrique, elle est pas contrôlée cette flamme, elle est jaune, "
            "orangée et elle va très loin. Le problème c'est qu'il faut baisser son intensité pour la ramener "
            "à une couleur bleue et cette flamme est maintenant très bien contrôlée, elle est sur une plus "
            "petite longueur et c'est avec ça qu'on va travailler en fait.\n\n"
            "J'allume le chalumeau avec un outil à arc électrique.\n\n"
            "En fait, j'utilise avec ma main droite l'outil à arc électrique et avec ma main gauche, j'ouvre "
            "le robinet du chalumeau pour faire passer un peu de gaz. Et quand j'allumerai avec le briquet, "
            "on est censé entendre un petit pop et la flamme s'allume d'un coup. Et c'est tout."
        ),
    },
    21: {
        "video_filename": "GX010229_etirage_coli_test_envoi.MP4",
        "start_time": 284.866,
        "end_time": 319.606,
        "duration": 34.74,
        "transcription": (
            "Alors, je prends dans ma main droite l'outil arc électrique. J'allume le chalumeau en augmentant "
            "la molette du dessus vers la gauche pour faire passer un peu plus de gaz. Je ne sais pas si c'est "
            "au gaz d'ailleurs. J'essaie de contrôler la flamme pour qu'elle soit moins incandescente jaune vif, "
            "mais plutôt bleue.\n\n"
            "Ma main gauche ne fait rien, ma main droite est la plus dominante, c'est elle qui utilise l'outil "
            "à arc électrique. Et lorsque l'on utilise la molette, elle est très facile à tourner, donc il faut "
            "bien manier son mouvement pour éviter des coups un peu très brusques."
        ),
    },
    22: {
        "video_filename": "Loic_biseauDroit.mov",
        "start_time": 1.071,
        "end_time": 40.867,
        "duration": 39.8,
        "transcription": (
            "J'amène le verre sur la meule pour piquer au centre de la trace de feutre. Je me mets au centre "
            "et je monte, je descends, je monte et je descends, je fais des aller-retours de haut en bas. Je "
            "vais de plus en plus loin et large jusqu'à venir mettre en pointe au croisement des traces de feutre. "
            "La mise en pointe le travail sur les extrémités du biseau, afin d'avoir une finesse des pointes du "
            "biseau. Il faut bien se placer à l'équerre, la surface du verre doit être perpendiculaire au plan "
            "de la meule, pour réaliser des pans réguliers et qui soient aussi épais à gauche et à droite. "
            "Autres choses sur ce point. Il faut y aller progressivement pour faire la mise en pointe et essayer "
            "d'être le plus régulier pour ne pas revenir ensuite une fois qu'on a terminé une pointe.\n\n"
            "Pas d'outil spécifique à ce stade. On tient le verre dans les mains. Le verre est tenu fermement "
            "entre les mains afin qu'il ne glisse pas et qu'il n'échappe pas lors de la taille.\n\n"
            "Je retiens ma respiration afin de ne pas créer de vibrations dans le verre qui se répercute dans "
            "la régularité de la taille du biseau.\n\n"
            "Je vois en transparence à travers le verre la progression du biseau dans l'épaisseur du verre, "
            "pour cela je regarde l'intérieur du verre.\n\n"
            "C'est ici l'aspect visuel qui va nous guider sur la régularité de l'épaisseur et de la profondeur "
            "du biseau. Encore une fois, ici c'est un marqueur visuel qui va nous permettre de vérifier le "
            "biseau, attention qu'il ne devienne pas irrégulier dans son épaisseur et sa largeur. Le ressenti "
            "est une pression régulière sur la meule et visuellement, la régularité de l'épaisseur du chanfrein "
            "est un signal d'un mouvement correct. Le son du verre sur la meule peut également être un indicateur, "
            "il doit lui aussi être régulier.\n\n"
            "Pour chaque biseau, le temps moyen est de trentaine de 30 à 45 secondes."
        ),
    },
    23: {
        "video_filename": "Loic_biseauOblique.mov.mp4",
        "start_time": 0.0,
        "end_time": 54.683,
        "duration": 54.68,
        "transcription": (
            "Le biseau oblique, le travail s'approche de celui pour le biseau droit. On se met au centre des "
            "tracés, on va se mettre en dedans, c'est-à-dire que l'on pose l'arête de la meule sur le tracé.  "
            "On commence à tourner le verre on le penche un petit peu vers la droite quand on va monter et "
            "inversement quand l'on redescend on le penche un peu à gauche, donc on pivote un peu le verre, "
            "vers la gauche ou vers la droite lors des mouvements descendant ou montant. Ici, j'ai fait un peu "
            "différemment. Je l'ai fait d'un seul coup en bas à gauche pour éviter de revenir plusieurs fois "
            "afin de faire une mise en pointe propre. Après, il y a plusieurs techniques. On peut aussi "
            "l'agrandir au fur et à mesure. Ça donne un biseau beaucoup plus régulier que lorsqu'on le fait "
            "tout de suite en l'agrandissant en bas.\n\n"
            "Pas d'outil spécifique à cette étape, on tient fermement le verre dans les mains, en veillant à "
            "garder de la souplesse au niveau des poignets, pour pouvoir manipuler le verre. Ici, la main droite "
            "soutient le verre par le dessous et la main gauche guide le verre. Les avant-bras sont posés sur "
            "une poutre que l'on aperçoit en bas de l'image, cela permet une bonne stabilité.\n\n"
            "Ici, le travail correct se voit par la régularité du biseau à travers l'épaisseur du verre. Un son "
            "régulier est également le marqueur d'une pression et d'une vitesse régulière sur la meule.\n\n"
            "Pour les apprentis qui débutent, l'écueil majeur est la régularité du biseau car leur geste ou la "
            "pression appliquée sur la meule, ne sont pas assez régulier. C'est également la gestuelle de bascule "
            "du verre sur la meule qui demande un apprentissage."
        ),
    },
}

# Course_83 docs (top 3 substantive ones)
COURSE_83_DOCS = [
    {
        "source": "course_83_module_1195_chunk_0",
        "content": (
            "Glassblowing is the practice of shaping a mass of glass that has been softened by heat by blowing "
            "air into it through a tube. Glassblowing was invented by Syrian craftsmen in the area of Sidon, Aleppo "
            "and later in Jerusalem around 50 BCE, and expanded widely throughout the Roman Empire."
        ),
    },
    {
        "source": "course_83_module_1187_chunk_0",
        "content": "Gesture Operational Model\n\nIllustrate the results of the analysis using GOM.",
    },
    {
        "source": "course_83_module_1186_chunk_0",
        "content": "Explain the installation in museum with photos, videos, etc.",
    },
]

ADVERSARIAL_QUERIES = [
    {"query": "Quelle est la capitale du Japon ?", "type": "off_topic"},
    {"query": "Comment réparer une voiture diesel ?", "type": "off_topic"},
    {"query": "bleu mardi rotation verre pourquoi", "type": "nonsense"},
    {"query": "Comment sculpter le marbre avec un ciseau à froid ?", "type": "adjacent_unanswerable"},
    {"query": "Quelle est la température de fusion de l'acier inoxydable ?", "type": "adjacent_unanswerable"},
]


def call_llm(prompt, retries=2):
    """Call Infomaniak LLM API."""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  LLM API error (attempt {attempt+1}): {e}")
            if attempt < retries:
                time.sleep(3)
    return None


def parse_json_array(text):
    """Parse a JSON array from LLM output, with fallback extraction."""
    if not text:
        return None
    # Try direct parse
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    # Find JSON array in text
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
    return None


def generate_annotation_queries(ann_id, transcription, register):
    """Generate 3 queries for one annotation and register."""
    if register == 'expert':
        prompt = f"""Étant donné cette annotation d'expert sur une technique artisanale (soufflage de verre) :
---
{transcription}
---
Génère 3 questions qu'un étudiant avec des connaissances techniques pourrait poser et auxquelles cette annotation répond directement. Utilise un vocabulaire technique précis correspondant au langage de l'annotation. Réponds avec un tableau JSON de 3 chaînes de caractères, sans explication."""
    else:
        prompt = f"""Étant donné cette annotation d'expert sur une technique artisanale (soufflage de verre) :
---
{transcription}
---
Génère 3 questions qu'un débutant complet pourrait poser et auxquelles cette annotation répond. Le débutant ne connaît PAS les termes techniques. Il décrit ce qu'il voit/ressent en français simple et familier. Par exemple : "pourquoi mon verre tombe" plutôt que "perte d'axialité de la paraison". Réponds avec un tableau JSON de 3 chaînes de caractères, sans explication."""

    print(f"  Generating {register} queries for annotation {ann_id}...")
    text = call_llm(prompt)
    queries = parse_json_array(text)

    if not queries or len(queries) < 3:
        print(f"  WARNING: Could not parse queries for ann {ann_id} {register}, using fallback")
        if register == 'expert':
            fallback = {
                17: [
                    "Comment utiliser l'arc électrique pour allumer le chalumeau en soufflage de verre ?",
                    "Quelle couleur indique une flamme bien contrôlée au chalumeau ?",
                    "Quelle est la procédure d'allumage du chalumeau avec le robinet à gaz ?",
                ],
                21: [
                    "Comment ajuster la molette du chalumeau pour contrôler le débit de gaz ?",
                    "Quelle est la technique d'allumage de l'arc électrique avec la main dominante ?",
                    "Comment éviter les mouvements brusques de la molette lors de l'allumage du chalumeau ?",
                ],
                22: [
                    "Quelle est la technique de mise en pointe du biseau droit sur la meule ?",
                    "Comment assurer la perpendicularité du verre sur le plan de la meule pour un biseau régulier ?",
                    "Quels sont les indicateurs sensoriels d'une taille correcte du biseau droit ?",
                ],
                23: [
                    "Comment réaliser le biseau oblique en pivotant le verre sur la meule ?",
                    "Quelle est la gestuelle de bascule du verre lors de la réalisation d'un biseau oblique ?",
                    "Comment la main droite et la main gauche se répartissent-elles lors de la taille du biseau oblique ?",
                ],
            }
        else:
            fallback = {
                17: [
                    "Comment on allume le feu pour travailler le verre ?",
                    "Pourquoi la flamme est d'abord jaune et ensuite bleue ?",
                    "C'est quoi ce bruit de pop qu'on entend quand on allume ?",
                ],
                21: [
                    "C'est quoi cette chose qu'on tourne pour faire passer le gaz ?",
                    "Pourquoi on utilise surtout la main droite pour le feu ?",
                    "Comment on fait pour ne pas faire partir trop de gaz d'un coup ?",
                ],
                22: [
                    "Comment on fait pour couper le verre en biais sans qu'il se casse ?",
                    "Pourquoi on retient sa respiration quand on travaille le verre sur la machine ?",
                    "Comment on sait si la coupe est régulière en regardant à travers le verre ?",
                ],
                23: [
                    "Comment on incline le verre pour faire une coupe pas droite mais en diagonale ?",
                    "Pourquoi c'est difficile de faire une belle coupe oblique pour les débutants ?",
                    "Comment tenir le verre des deux mains pour le couper en oblique ?",
                ],
            }
        queries = fallback.get(ann_id, [f"Question {i+1} sur annotation {ann_id}" for i in range(3)])

    return queries[:3]


def generate_course_queries(doc, doc_idx):
    """Generate 3 queries for one course doc."""
    prompt = f"""Étant donné ce contenu de cours sur le soufflage de verre :
---
{doc['content']}
---
Génère 3 questions auxquelles ce contenu répond directement. Réponds avec un tableau JSON de 3 chaînes de caractères, sans explication."""

    print(f"  Generating queries for course_83 doc {doc_idx}...")
    text = call_llm(prompt)
    queries = parse_json_array(text)

    if not queries or len(queries) < 3:
        print(f"  WARNING: Could not parse course queries for doc {doc_idx}, using fallback")
        fallbacks = [
            [
                "What is glassblowing and where was it invented ?",
                "Quand a été inventé le soufflage de verre ?",
                "Comment le soufflage de verre s'est répandu dans l'Empire romain ?",
            ],
            [
                "Qu'est-ce que le modèle opérationnel gestuel (GOM) en soufflage de verre ?",
                "Comment utilise-t-on le GOM pour analyser les gestes en verrerie ?",
                "Quels résultats peut-on illustrer avec le modèle GOM ?",
            ],
            [
                "Comment documenter une installation de verrerie dans un musée ?",
                "Quels médias utilise-t-on pour expliquer une installation de soufflage de verre ?",
                "Pourquoi utiliser des photos et vidéos pour documenter une installation artisanale ?",
            ],
        ]
        queries = fallbacks[doc_idx % len(fallbacks)]

    return queries[:3]


def build_ground_truth():
    ground_truth = []
    qid_counter = {}

    def next_qid(prefix):
        qid_counter[prefix] = qid_counter.get(prefix, 0) + 1
        return f"{prefix}_{qid_counter[prefix]}"

    # 1. Annotation queries (expert + novice) for each of 4 annotations
    for ann_id, ann_data in ANNOTATIONS.items():
        transcription = ann_data['transcription']
        source_key = f"{ann_data['video_filename']}#{ann_id}_raw"

        for register in ['expert', 'novice']:
            queries = generate_annotation_queries(ann_id, transcription, register)
            time.sleep(1)  # Rate limiting

            for i, q in enumerate(queries):
                qid = f"ann{ann_id}_{register}_{i+1}"
                entry = {
                    "qid": qid,
                    "query": q,
                    "register": register,
                    "source": "annotation",
                    "relevant_annotation_ids": [ann_id],
                    "relevant_source_keys": [source_key],
                    "generated_from": f"annotation_{ann_id}",
                }
                ground_truth.append(entry)
                print(f"    [{qid}] {q}")

    # 2. Course content queries for top 3 course_83 docs
    print("\nGenerating course queries...")
    # Get actual course_83 docs from ChromaDB
    import chromadb
    chroma_client = chromadb.PersistentClient(path='./chroma_langchain_db')
    c83 = chroma_client.get_collection('course_83')
    c83_results = c83.get(limit=3, include=['documents', 'metadatas'])

    for doc_idx, (doc_text, meta) in enumerate(zip(c83_results['documents'], c83_results['metadatas'])):
        doc_source = meta.get('source', f'course_83_doc_{doc_idx}')
        doc = {"source": doc_source, "content": doc_text}
        queries = generate_course_queries(doc, doc_idx)
        time.sleep(1)

        for i, q in enumerate(queries):
            qid = f"course83_{doc_idx * 3 + i + 1}"
            entry = {
                "qid": qid,
                "query": q,
                "register": "expert",
                "source": "course_content",
                "relevant_annotation_ids": [],
                "relevant_source_keys": [],
                "relevant_course_sources": [doc_source],
                "generated_from": f"course_83_doc_{doc_idx}",
            }
            ground_truth.append(entry)
            print(f"    [{qid}] {q}")

    # 3. Adversarial queries
    print("\nAdding adversarial queries...")
    for i, adv in enumerate(ADVERSARIAL_QUERIES):
        qid = f"adv_{i+1}"
        entry = {
            "qid": qid,
            "query": adv['query'],
            "register": "adversarial",
            "source": "adversarial",
            "relevant_annotation_ids": [],
            "relevant_source_keys": [],
            "adversarial_type": adv['type'],
        }
        ground_truth.append(entry)
        print(f"    [{qid}] {adv['query']} ({adv['type']})")

    return ground_truth


def main():
    print("=" * 60)
    print("STEP 3: Generate Synthetic Queries")
    print("=" * 60)
    print(f"LLM: {LLM_MODEL}")
    print()

    ground_truth = build_ground_truth()

    out_path = os.path.join(FIXTURES_DIR, 'ground_truth.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(ground_truth)} queries to {out_path}")

    # Summary
    by_register = {}
    for e in ground_truth:
        r = e['register']
        by_register[r] = by_register.get(r, 0) + 1

    print("\nQuery distribution:")
    for register, count in sorted(by_register.items()):
        print(f"  {register}: {count}")

    print("=" * 60)
    print("COMPLETE: Query generation done.")
    print("=" * 60)


if __name__ == '__main__':
    main()
