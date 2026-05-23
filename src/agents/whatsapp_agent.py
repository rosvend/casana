"""WhatsApp outreach agent — verifies listing availability via EvolutionAPI.

Triggered by ``evaluation_router`` after ``EvaluationResult.passes`` is True.
Reads scored ``candidates``, picks the top ones (``match_score >= 0.70``,
capped at the top 3), sends each broker a short Colombian Spanish message via
EvolutionAPI's ``/message/sendText`` endpoint, polls ``/chat/findMessages``
for ~30 s to detect a reply, and promotes every candidate's ``Listing`` to a
``VerifiedListing`` (candidates not contacted are still promoted so the
downstream schema stays uniform).

The node is a no-op when the user has disabled outreach. The state field
``whatsapp_enabled`` wins when present; otherwise the ``WHATSAPP_ENABLED``
env var is honored (default off, so tests and dev runs never reach a real
broker).
"""

from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from src.state import Candidate, Listing, PropertyFinderState, VerifiedListing

load_dotenv()

logger = logging.getLogger(__name__)

MIN_SCORE = 0.70
TOP_N = 3
POLL_INTERVAL_SECONDS = 5
POLL_MAX_ATTEMPTS = 6
JITTER_RANGE = (3, 8)
OUTREACH_TEXT = (
    "Hola, vi esta propiedad en internet y estoy interesado. "
    "¿Aún está disponible?"
)


def whatsapp_node(state: PropertyFinderState) -> dict:
    """Contact brokers for the top candidates and promote listings to verified.

    Returns ``{"candidates": [...]}`` — every input candidate appears in the
    output exactly once, with its ``listing`` field replaced by a
    ``VerifiedListing`` recording the outreach outcome.
    """
    candidates = state.get("candidates") or []
    if not candidates:
        logger.info("whatsapp_node: no candidates, nothing to verify")
        return {}

    if not _outreach_enabled(state):
        logger.info(
            "whatsapp_node: outreach disabled, promoting candidates without contact"
        )
        return {
            "candidates": [
                _promote(c, replied=False, notes="WhatsApp outreach disabled for this run.")
                for c in candidates
            ]
        }

    selected_ids = _select_candidate_ids(candidates)

    base_url, api_key, instance = _evolution_config()
    config_ok = bool(base_url and api_key and instance)
    if not config_ok:
        logger.warning(
            "whatsapp_node: EvolutionAPI not configured "
            "(EVOLUTION_API_URL/EVOLUTION_API_KEY/EVOLUTION_INSTANCE) — skipping outreach"
        )

    updated: list[Candidate] = []
    for candidate in candidates:
        if candidate.listing.id not in selected_ids:
            updated.append(
                _promote(
                    candidate,
                    replied=False,
                    notes=(
                        f"Not contacted: match_score {candidate.match_score:.2f} "
                        f"below 0.70 threshold or outside top {TOP_N}."
                    ),
                )
            )
            continue

        phone = _first_phone(candidate.listing)
        if phone is None:
            updated.append(
                _promote(
                    candidate,
                    replied=False,
                    notes="Not contacted: no phone numbers extracted from listing.",
                )
            )
            continue

        if not config_ok:
            updated.append(
                _promote(
                    candidate,
                    replied=False,
                    notes="Not contacted: EvolutionAPI environment variables missing.",
                )
            )
            continue

        digits = _format_number(phone)
        send_time = int(time.time())
        sent_ok = _send_text(base_url, api_key, instance, digits)
        if not sent_ok:
            updated.append(
                _promote(
                    candidate,
                    replied=False,
                    notes=f"Send failed for {digits}; broker not reached.",
                )
            )
            time.sleep(random.uniform(*JITTER_RANGE))
            continue

        replied = _poll_for_reply(base_url, api_key, instance, digits, send_time)
        notes = (
            "Broker confirmed availability via WhatsApp."
            if replied
            else f"No reply within {POLL_INTERVAL_SECONDS * POLL_MAX_ATTEMPTS}s of outreach."
        )
        updated.append(_promote(candidate, replied=replied, notes=notes))
        time.sleep(random.uniform(*JITTER_RANGE))

    return {"candidates": updated}


def _outreach_enabled(state: PropertyFinderState) -> bool:
    if "whatsapp_enabled" in state:
        return bool(state["whatsapp_enabled"])
    return os.getenv("WHATSAPP_ENABLED", "false").strip().lower() == "true"


def _evolution_config() -> tuple[str, str, str]:
    return (
        os.getenv("EVOLUTION_API_URL", "").rstrip("/"),
        os.getenv("EVOLUTION_API_KEY", ""),
        os.getenv("EVOLUTION_INSTANCE", ""),
    )


def _select_candidate_ids(candidates: list[Candidate]) -> set[str]:
    eligible = [c for c in candidates if c.match_score >= MIN_SCORE]
    eligible.sort(key=lambda c: c.match_score, reverse=True)
    return {c.listing.id for c in eligible[:TOP_N]}


def _first_phone(listing: Listing) -> str | None:
    for raw in listing.phone_numbers:
        if raw and raw.strip():
            return raw.strip()
    return None


def _format_number(raw: str) -> str:
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) == 10 and not digits.startswith("57"):
        digits = "57" + digits
    return digits


def _send_text(base_url: str, api_key: str, instance: str, number: str) -> bool:
    url = f"{base_url}/message/sendText/{instance}"
    try:
        response = requests.post(
            url,
            headers={"apikey": api_key, "Content-Type": "application/json"},
            json={"number": number, "text": OUTREACH_TEXT},
            timeout=10,
        )
    except requests.RequestException as exc:
        logger.warning("whatsapp_node: send failed for %s: %s", number, exc)
        return False
    if response.status_code >= 300:
        logger.warning(
            "whatsapp_node: send returned %s for %s", response.status_code, number
        )
        return False
    return True


def _poll_for_reply(
    base_url: str, api_key: str, instance: str, number: str, send_time: int
) -> bool:
    """POST to Evolution's findMessages, looking for an inbound reply since send_time.

    Filters server-side on ``fromMe: false`` only. WhatsApp's LID system means
    inbound replies can arrive under a ``...@lid`` ``remoteJid`` with the phone
    JID surfaced as ``key.remoteJidAlt`` — so the server filter must NOT pin a
    specific remoteJid; ``_has_inbound_reply`` does the JID match client-side
    across both forms.
    """
    url = f"{base_url}/chat/findMessages/{instance}"
    body = {"where": {"key": {"fromMe": False}}}
    for _ in range(POLL_MAX_ATTEMPTS):
        try:
            response = requests.post(
                url,
                headers={"apikey": api_key, "Content-Type": "application/json"},
                json=body,
                timeout=10,
            )
        except requests.RequestException as exc:
            logger.warning("whatsapp_node: poll failed for %s: %s", number, exc)
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        if response.status_code < 300 and _has_inbound_reply(response, number, send_time):
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    return False


def _has_inbound_reply(response, number: str, send_time: int) -> bool:
    """True if the response contains an inbound message from `number` newer than send_time.

    Evolution 2.3.x returns ``{"messages": {"records": [{"key": {"fromMe": ..., "remoteJid": ...}, "messageTimestamp": ...}, ...]}}``.
    """
    try:
        body = response.json()
    except Exception:
        return False
    if not isinstance(body, dict):
        return False
    messages_wrap = body.get("messages")
    if isinstance(messages_wrap, dict):
        records = messages_wrap.get("records") or []
    elif isinstance(messages_wrap, list):
        records = messages_wrap
    else:
        return False
    for msg in records:
        if not isinstance(msg, dict):
            continue
        key = msg.get("key") if isinstance(msg.get("key"), dict) else {}
        if key.get("fromMe") is True or msg.get("fromMe") is True:
            continue
        jid_candidates = [
            str(key.get("remoteJid", "")),
            str(key.get("remoteJidAlt", "")),
            str(msg.get("from", "")),
        ]
        if not any(number in j for j in jid_candidates if j):
            continue
        ts = msg.get("messageTimestamp") or msg.get("timestamp") or 0
        try:
            ts_int = int(ts)
        except (TypeError, ValueError):
            ts_int = 0
        if ts_int >= send_time:
            return True
    return False


def _promote(candidate: Candidate, *, replied: bool, notes: str) -> Candidate:
    base = candidate.listing
    base_fields = {
        k: v for k, v in base.model_dump().items() if k in Listing.model_fields
    }
    verified = VerifiedListing(
        **base_fields,
        availability_confirmed=replied,
        verification_timestamp=datetime.now(timezone.utc),
        verification_notes=notes,
    )
    return candidate.model_copy(update={"listing": verified})
