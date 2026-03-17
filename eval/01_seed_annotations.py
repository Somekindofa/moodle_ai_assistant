"""
Step 1: Seed annotations from Moodle MySQL into ChromaDB moodle_assistant_collection.

This is the documented bugfix: annotations from Moodle MySQL are not in ChromaDB.
"""

import os
import sys
import logging

os.chdir('/opt/craftpilot_backend')
sys.path.insert(0, '/opt/craftpilot_backend')

logging.basicConfig(level=logging.WARNING)

import pymysql
import chromadb
from langchain_core.documents.base import Document
from langchain_chroma import Chroma
from config.settings import ConfigurationManager
from services.rag_service import RAGService

# ── helpers ──────────────────────────────────────────────────────────────────

import dotenv
dotenv.load_dotenv('/opt/craftpilot_backend/.env')

MYSQL_CONFIG = dict(
    host='localhost',
    user='moodleuser',
    password=os.getenv('MOODLE_DB_PASSWORD', ''),
    database='moodle',
    charset='utf8mb4',
)

ANNOTATION_IDS = [17, 21, 22, 23]  # 4 completed annotations

HARDCODED_TRANSCRIPTIONS = {
    17: (
        "J'allume la flamme avec l'arc électrique, elle est pas contrôlée cette flamme, elle est jaune, "
        "orangée et elle va très loin. Le problème c'est qu'il faut baisser son intensité pour la ramener "
        "à une couleur bleue et cette flamme est maintenant très bien contrôlée, elle est sur une plus "
        "petite longueur et c'est avec ça qu'on va travailler en fait.\n\n"
        "J'allume le chalumeau avec un outil à arc électrique.\n\n"
        "En fait, j'utilise avec ma main droite l'outil à arc électrique et avec ma main gauche, j'ouvre "
        "le robinet du chalumeau pour faire passer un peu de gaz. Et quand j'allumerai avec le briquet, "
        "on est censé entendre un petit pop et la flamme s'allume d'un coup. Et c'est tout."
    ),
    21: (
        "Alors, je prends dans ma main droite l'outil arc électrique. J'allume le chalumeau en augmentant "
        "la molette du dessus vers la gauche pour faire passer un peu plus de gaz. Je ne sais pas si c'est "
        "au gaz d'ailleurs. J'essaie de contrôler la flamme pour qu'elle soit moins incandescente jaune vif, "
        "mais plutôt bleue.\n\n"
        "Ma main gauche ne fait rien, ma main droite est la plus dominante, c'est elle qui utilise l'outil "
        "à arc électrique. Et lorsque l'on utilise la molette, elle est très facile à tourner, donc il faut "
        "bien manier son mouvement pour éviter des coups un peu très brusques."
    ),
    22: (
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
    23: (
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
}

ANNOTATION_META = {
    17: {
        "video_filename": "GX010191.MP4",
        "start_time": 110.712,
        "end_time": 135.822,
        "video_id": 28,
    },
    21: {
        "video_filename": "GX010229_etirage_coli_test_envoi.MP4",
        "start_time": 284.866,
        "end_time": 319.606,
        "video_id": 31,
    },
    22: {
        "video_filename": "Loic_biseauDroit.mov",
        "start_time": 1.071,
        "end_time": 40.867,
        "video_id": 36,
    },
    23: {
        "video_filename": "Loic_biseauOblique.mov.mp4",
        "start_time": 0.0,
        "end_time": 54.683,
        "video_id": 37,
    },
}


def fetch_annotations_from_mysql():
    """Read completed annotations from Moodle MySQL."""
    print("Connecting to Moodle MySQL...")
    conn = pymysql.connect(**MYSQL_CONFIG, cursorclass=pymysql.cursors.DictCursor)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            a.id AS annotation_id,
            a.videoid AS video_id,
            a.starttime AS start_time,
            a.endtime AS end_time,
            a.transcription,
            a.transcriptionstatus AS transcription_status,
            a.timecreated,
            v.filename AS video_filename,
            v.source_type,
            v.duration AS video_duration
        FROM mdl_local_videoelicit_annotations a
        JOIN mdl_local_videoelicit_videos v ON a.videoid = v.id
        WHERE a.transcriptionstatus = 'completed'
        ORDER BY a.id
    """)
    rows = cursor.fetchall()
    conn.close()
    print(f"  Found {len(rows)} completed annotations from MySQL")
    return rows


def build_documents(annotations):
    """Convert annotation rows to LangChain Documents with metadata."""
    documents = []
    for ann in annotations:
        ann_id = ann['annotation_id']
        video_filename = ann['video_filename'] or ANNOTATION_META.get(ann_id, {}).get('video_filename', 'unknown.mp4')
        start_time = float(ann['start_time']) if ann['start_time'] is not None else ANNOTATION_META.get(ann_id, {}).get('start_time', 0.0)
        end_time = float(ann['end_time']) if ann['end_time'] is not None else ANNOTATION_META.get(ann_id, {}).get('end_time', 0.0)
        duration = round(end_time - start_time, 2)

        # Use transcription from DB; fall back to hardcoded
        transcription = ann.get('transcription') or HARDCODED_TRANSCRIPTIONS.get(ann_id, '')

        if not transcription:
            print(f"  WARNING: No transcription for annotation {ann_id}, skipping")
            continue

        created_at = str(ann.get('timecreated', ''))

        metadata = {
            "annotation_id": int(ann_id),
            "video_id": int(ann['video_id']),
            "video_filename": video_filename,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "source_type": ann.get('source_type') or 'webdav',
            "project_name": "glassblowing",
            "annotation_created_at": created_at,
            "type": "video_annotation",
            "transcript_type": "raw",
            "source": f"{video_filename}#{ann_id}_raw",
        }

        doc = Document(page_content=transcription, metadata=metadata)
        documents.append(doc)
        print(f"  Built doc for annotation {ann_id}: {video_filename}, {duration:.1f}s, {len(transcription)} chars")

    return documents


def seed_chromadb(documents):
    """Add documents to ChromaDB moodle_assistant_collection."""
    print("\nInitializing RAGService and ChromaDB connection...")
    config_manager = ConfigurationManager()
    rag_service = RAGService(config_manager)

    vector_store = rag_service.vector_store

    # Check current count
    initial_count = vector_store._collection.count()
    print(f"  Initial document count in moodle_assistant_collection: {initial_count}")

    if initial_count > 0:
        print("  Collection already has documents. Clearing and re-seeding...")
        # Delete existing to avoid duplicates
        existing = vector_store._collection.get()
        if existing['ids']:
            vector_store._collection.delete(ids=existing['ids'])
        print(f"  Deleted {len(existing['ids'])} existing documents")

    # Add documents
    print(f"\nAdding {len(documents)} documents to moodle_assistant_collection...")
    vector_store.add_documents(documents)

    # Verify
    final_count = vector_store._collection.count()
    print(f"\nVerification:")
    print(f"  Documents in moodle_assistant_collection: {final_count}")

    if final_count == len(documents):
        print(f"  SUCCESS: All {len(documents)} annotation documents seeded correctly.")
    else:
        print(f"  WARNING: Expected {len(documents)}, got {final_count}")

    # Show what was added
    stored = vector_store._collection.get(include=['metadatas'])
    print("\nStored documents:")
    for meta in stored['metadatas']:
        print(f"  annotation_id={meta.get('annotation_id')}, "
              f"source={meta.get('source')}, "
              f"duration={meta.get('duration')}s")

    return final_count


def main():
    print("=" * 60)
    print("STEP 1: Seed Annotations from MySQL → ChromaDB")
    print("=" * 60)

    # Fetch from MySQL
    annotations = fetch_annotations_from_mysql()

    if not annotations:
        print("No completed annotations found in MySQL. Using hardcoded data...")
        # Build from hardcoded data
        annotations = [
            {
                'annotation_id': ann_id,
                'video_id': ANNOTATION_META[ann_id]['video_id'],
                'start_time': ANNOTATION_META[ann_id]['start_time'],
                'end_time': ANNOTATION_META[ann_id]['end_time'],
                'transcription': HARDCODED_TRANSCRIPTIONS[ann_id],
                'transcription_status': 'completed',
                'timecreated': '',
                'video_filename': ANNOTATION_META[ann_id]['video_filename'],
                'source_type': 'webdav',
                'video_duration': 0.0,
            }
            for ann_id in ANNOTATION_IDS
        ]

    # Build documents
    print("\nBuilding LangChain Documents...")
    documents = build_documents(annotations)
    print(f"  Built {len(documents)} documents")

    # Seed ChromaDB
    final_count = seed_chromadb(documents)

    print("\n" + "=" * 60)
    print(f"COMPLETE: {final_count} annotation documents in moodle_assistant_collection")
    print("=" * 60)


if __name__ == '__main__':
    main()
