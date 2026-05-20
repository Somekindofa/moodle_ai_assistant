"""
Session summary service.

One-shot Infomaniak LLM call that, given an elicitation transcript and the
per-phase coverage scores from the spaCy detector, returns a short French
synthesis plus 2-3 follow-up questions targeting the weakest phase.

Termination is decided BEFORE this service is called: if the coverage
detector's plateau gate fires, the caller skips this call entirely. When this
service IS called, it is free to return an empty `follow_ups` list if the
transcript genuinely shows no gap — but under normal use the gate has already
filtered out those cases.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

logger = logging.getLogger(__name__)

# Model: mistral3 is the cheaper/faster of the valid Infomaniak IDs and
# suffices for constrained JSON synthesis. Apertus-70B is overkill here.
_SUMMARY_MODEL = "mistral3"

_SYSTEM_PROMPT = """Tu analyses une session d'élicitation d'un artisan qui décrit une activité qu'il vient d'effectuer.

La session est structurée en trois phases :
- QUOI : description observable des gestes, outils, objets manipulés.
- COMMENT : manière, séquence, vitesse, outils utilisés, modulations.
- POURQUOI : intentions, causes, objectifs, erreurs à éviter, raisons des choix.

Ton rôle :
1. Produire un résumé de 2 à 4 phrases de ce qui a été dit. Reste descriptif, ne juge pas.
2. Identifier la phase la PLUS sous-développée d'après la transcription (quoi, comment, pourquoi), ou null si les trois sont bien couvertes.
3. Proposer 2 à 3 questions de relance courtes, concrètes, en tutoiement, pour approfondir cette phase. Renvoyer une liste vide si la session est complète et qu'il n'y a rien de substantiel à approfondir.

RÈGLES STRICTES :
- N'invente rien. Si tu cites un geste, un outil ou un objet, il doit apparaître explicitement dans la transcription.
- Si la transcription est trop pauvre pour une question concrète, renvoie une liste vide plutôt qu'une question générique.
- Réponds UNIQUEMENT en JSON valide, sans texte avant ni après, selon le schéma exact :
{"summary": "...", "weakest_phase": "quoi" | "comment" | "pourquoi" | null, "follow_ups": ["...", "..."]}
"""


def _build_llm() -> ChatOpenAI:
    """Non-streaming ChatOpenAI pointed at Infomaniak. Low temperature for JSON stability."""
    api_key = os.getenv("INFOMANIAK_API_KEY", "")
    product_id = os.getenv("INFOMANIAK_PRODUCT_ID", "")
    if not api_key or not product_id:
        raise RuntimeError("INFOMANIAK_API_KEY / INFOMANIAK_PRODUCT_ID not set")
    base_url = f"https://api.infomaniak.com/2/ai/{product_id}/openai/v1"
    return ChatOpenAI(
        model=_SUMMARY_MODEL,
        openai_api_key=api_key,
        openai_api_base=base_url,
        streaming=False,
        temperature=0.2,
        max_tokens=600,
    )


def _format_score_hint(phase_scores: dict[str, Any]) -> str:
    """Render the aggregate coverage dict as a compact hint for the prompt."""
    lines = []
    for phase in ("quoi", "comment", "pourquoi"):
        s = phase_scores.get(phase, {}) or {}
        hits = s.get("hits", 0)
        status = s.get("status", "absent")
        lines.append(f"- {phase} : {hits} marqueurs ({status})")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction — handles code fences or stray prose."""
    # Try direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to first {...} block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in LLM output: {text[:200]!r}")
    return json.loads(m.group(0))


@traceable(name="summarize_elicitation")
async def summarize_elicitation(
    transcript: str,
    phase_scores: dict[str, Any],
) -> dict[str, Any]:
    """
    Call Infomaniak once to produce a summary + targeted follow-ups.

    Returns a dict with keys: summary (str), weakest_phase (str|None),
    follow_ups (list[str]). Caller validates shape via pydantic.
    """
    if not transcript.strip():
        return {"summary": "", "weakest_phase": None, "follow_ups": []}

    llm = _build_llm()
    user_msg = (
        f"Transcription de la session :\n<<<\n{transcript.strip()}\n>>>\n\n"
        f"Scores de couverture (calculés hors-LLM par un détecteur de marqueurs) :\n"
        f"{_format_score_hint(phase_scores)}\n\n"
        f"Produis le JSON demandé."
    )

    response = await llm.ainvoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])
    raw = response.content if isinstance(response.content, str) else str(response.content)

    parsed = _extract_json(raw)

    # Normalise shape — the LLM sometimes returns weakest_phase as a string
    # inside a longer sentence, or forgets the key entirely.
    summary = (parsed.get("summary") or "").strip()
    weakest = parsed.get("weakest_phase")
    if isinstance(weakest, str):
        w = weakest.strip().lower()
        weakest = w if w in {"quoi", "comment", "pourquoi"} else None
    else:
        weakest = None
    follow_ups = parsed.get("follow_ups") or []
    if not isinstance(follow_ups, list):
        follow_ups = []
    follow_ups = [str(q).strip() for q in follow_ups if str(q).strip()][:3]

    return {"summary": summary, "weakest_phase": weakest, "follow_ups": follow_ups}
