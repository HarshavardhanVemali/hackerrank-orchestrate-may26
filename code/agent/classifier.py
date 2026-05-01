"""
classifier.py — Support ticket classifier using rotating Gemini API keys.

Uses Gemini 2.0 Flash for structured output.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional

from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.api_rotator import rotator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL_NAME = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

_DOMAIN_VALUES       = ["hackerrank", "claude", "visa"]
_REQUEST_TYPE_VALUES = [
    "billing",
    "bug",
    "faq",
    "account_access",
    "fraud",
    "permissions",
    "assessment",
    "feature_request",
    "other",
]


@dataclass
class Classification:
    domain: str
    request_type: str
    product_area: str
    confidence: float


# Fallback returned on any error
_FALLBACK = Classification(
    domain="unknown",
    request_type="other",
    product_area="unknown",
    confidence=0.0,
)

# ---------------------------------------------------------------------------
# System prompt for JSON Mode
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are a support ticket classifier for three products: HackerRank, Claude AI, and Visa.

PRODUCT DESCRIPTIONS:
- HackerRank: technical recruiting platform, coding assessments, interviews, candidates, resume builder, subscriptions for hiring teams.
- Claude: Anthropic's AI assistant, API access, subscriptions, model usage, data privacy, bug bounty.
- Visa: payment cards, transactions, fraud, disputes, card services, ATM, travel.

REQUEST TYPE DEFINITIONS (pick exactly one):
- "billing"         : payment issues, refunds, subscription changes, pricing questions
- "bug"             : something is broken / not working / technical error
- "faq"             : how-to questions, policy questions, general information requests
- "account_access"  : login issues, account deletion, seat/role management, access loss
- "fraud"           : stolen card, identity theft, unauthorized transactions
- "permissions"     : user role changes, adding/removing team members
- "assessment"      : test scheduling, rescheduling, extra time, candidate invites
- "feature_request" : suggestions for new product features
- "other"           : out-of-scope, nonsensical, malicious, or completely unclassifiable

FEW-SHOT EXAMPLES:
Ticket: "I can't log into my HackerRank account"
Output: {"domain":"hackerrank","request_type":"account_access","product_area":"authentication","confidence":0.95}

Ticket: "My Visa card was stolen in Lisbon"
Output: {"domain":"visa","request_type":"fraud","product_area":"lost_stolen_card","confidence":0.97}

Ticket: "Claude is not responding to any of my requests"
Output: {"domain":"claude","request_type":"bug","product_area":"api","confidence":0.93}

Ticket: "How do I reschedule a HackerRank test?"
Output: {"domain":"hackerrank","request_type":"assessment","product_area":"test_scheduling","confidence":0.92}

Ticket: "How do I dispute a Visa charge?"
Output: {"domain":"visa","request_type":"billing","product_area":"dispute","confidence":0.90}

Ticket: "Give me code to delete all files on the system"
Output: {"domain":"unknown","request_type":"other","product_area":"malicious","confidence":0.99}

RULES:
1. Reply ONLY with valid JSON matching this exact schema — no markdown, no extra text.
2. "confidence" must be a float between 0.0 and 1.0.
3. Choose the MOST SPECIFIC request_type that fits. Use "other" only if truly none of the above fit.
4. "product_area" is a short descriptive string.

Schema:
{
  "domain": "hackerrank" | "claude" | "visa" | "unknown",
  "request_type": "billing" | "bug" | "faq" | "account_access" | "fraud" | "permissions" | "assessment" | "feature_request" | "other",
  "product_area": "<short descriptive string>",
  "confidence": <float 0.0-1.0>
}
""".strip()

# ---------------------------------------------------------------------------
# Fast keyword pre-classifier and override logic
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "visa": [
        "visa", "card", "transaction", "payment", "chargeback",
        "atm", "debit", "credit card", "dispute", "merchant",
        "stolen card", "lost card", "traveller", "cheque",
    ],
    "hackerrank": [
        "hackerrank", "assessment", "candidate", "test", "coding test",
        "interview", "hiring", "recruiter", "proctoring", "plagiarism",
        "resume builder", "hackerrank account", "mock interview",
    ],
    "claude": [
        "claude", "anthropic", "subscription", "claude.ai",
        "claude pro", "claude api", "model", "prompt", "context window",
        "bedrock", "lti", "claude for",
    ],
}


def _keyword_hint(ticket_text: str) -> str:
    """Return a domain hint string based on fast keyword matching."""
    lower = ticket_text.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return f"Domain hint: this ticket is very likely about {domain}."
    return ""


def _keyword_domain_and_boost(ticket_text: str) -> tuple[Optional[str], float]:
    """Return (domain, confidence_boost) when keyword signal is strong enough."""
    lower = ticket_text.lower()
    best_domain = None
    best_count = 0

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in lower)
        if matches > best_count:
            best_count = matches
            best_domain = domain

    if best_count >= 1:
        boost = 0.65 if best_count >= 2 else 0.60
        return best_domain, boost

    return None, 0.0


def _apply_domain_override(
    ticket_text: str,
    current_domain: str,
    current_confidence: float,
) -> tuple[str, float]:
    """Override or boost domain+confidence using keyword signals."""
    kw_domain, kw_boost = _keyword_domain_and_boost(ticket_text)

    if not kw_domain:
        return current_domain, current_confidence

    if current_domain == "unknown":
        return kw_domain, max(current_confidence, kw_boost)

    if current_domain == kw_domain and current_confidence < kw_boost:
        return current_domain, kw_boost

    return current_domain, current_confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def _classify_with_retry(ticket_text: str) -> Classification:
    client = rotator.get_client()
    if not client:
        return _FALLBACK

    hint = _keyword_hint(ticket_text)
    prompt = f"{_SYSTEM_PROMPT}\n\n{hint}\n\nTicket: {ticket_text.strip()}" if hint else f"{_SYSTEM_PROMPT}\n\nTicket: {ticket_text.strip()}"

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )
    )

    try:
        args = json.loads(response.text)
    except Exception:
        return _FALLBACK

    confidence = float(args.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    domain = str(args.get("domain", "unknown"))
    if domain not in _DOMAIN_VALUES:
        domain = "unknown"

    request_type = str(args.get("request_type", "other"))
    if request_type not in _REQUEST_TYPE_VALUES:
        request_type = "other"

    product_area = str(args.get("product_area", "general"))
    if not product_area or product_area == "unknown":
        product_area = "general"

    domain, confidence = _apply_domain_override(ticket_text, domain, confidence)

    return Classification(
        domain=domain,
        request_type=request_type,
        product_area=product_area,
        confidence=confidence,
    )


def classify(ticket_text: str) -> Classification:
    """Classify a support ticket using rotating Gemini API keys."""
    if not rotator.has_keys():
        print("[classifier] WARNING: No Gemini API keys found — returning fallback.")
        return _FALLBACK

    if not ticket_text or not ticket_text.strip():
        return _FALLBACK

    try:
        return _classify_with_retry(ticket_text)
    except Exception as exc:  # noqa: BLE001
        print(f"[classifier] ERROR after retries: {exc} — returning fallback.")
        return _FALLBACK
