"""
responder.py — Support ticket responder using rotating Gemini API keys.

Responsible for:
  - Building grounded prompts from retrieved documents
  - Calling Gemini 2.0 Flash to generate replies
  - Hallucination checks
"""

import re
from dataclasses import dataclass, field
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.classifier import Classification
from agent.safety import SafetyDecision
from corpus import loader
from corpus.loader import Document
from utils.api_rotator import rotator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL_NAME = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    action: str
    response: str
    sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CORPUS_NOT_FOUND = (
    "I don't have enough information in our support documentation to answer this. "
    "Please contact our support team directly."
)

_MIN_OVERLAP_WORDS = 3

# Regex patterns for PII detection
_EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
_PHONE_PATTERN = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"


def _get_top_score(query: str, docs: list[Document]) -> float:
    """Helper to check the relative quality of search results (using word overlap)."""
    if not docs: return 0.0
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    doc_words = set(re.findall(r"[a-z0-9]+", docs[0].content.lower() + " " + docs[0].title.lower()))
    return len(query_words & doc_words)


def _check_hallucination(
    reply_text: str,
    retrieved_docs: list[Document],
) -> list[str]:
    """Flag response sentences with low lexical overlap or PII leaks."""
    corpus_words: set[str] = set()
    corpus_raw: str = ""
    for doc in retrieved_docs:
        lower_content = doc.content.lower()
        lower_title = doc.title.lower()
        corpus_words.update(re.findall(r"[a-z0-9]+", lower_content))
        corpus_words.update(re.findall(r"[a-z0-9]+", lower_title))
        corpus_raw += lower_content + " " + lower_title

    if not corpus_words:
        return []

    # 1. PII Check: Look for emails/phones in response that are NOT in the corpus
    response_emails = re.findall(_EMAIL_PATTERN, reply_text)
    response_phones = re.findall(_PHONE_PATTERN, reply_text)
    
    pii_violations = []
    for email in response_emails:
        if email.lower() not in corpus_raw:
            pii_violations.append(f"Unauthorized Email Leak: {email}")
            
    for phone in response_phones:
        # Normalize phone for check (rough)
        clean_phone = re.sub(r"\D", "", phone)
        if clean_phone not in re.sub(r"\D", "", corpus_raw):
            pii_violations.append(f"Unauthorized Phone Leak: {phone}")

    # 2. Lexical Overlap Check
    sentences = re.split(r"(?<=[.!?])\s+", reply_text.strip())
    flagged: list[str] = pii_violations

    for sentence in sentences:
        if len(sentence.strip()) < 25: # Slightly stricter limit
            continue
        sentence_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        overlap = sentence_words & corpus_words
        if len(overlap) < _MIN_OVERLAP_WORDS:
            flagged.append(f"Low Overlap: {sentence.strip()}")

    return flagged


def _build_context(docs: list[Document]) -> tuple[str, list[str]]:
    """Format retrieved documents into a numbered context block."""
    blocks: list[str] = []
    source_urls: list[str] = []

    for i, doc in enumerate(docs, start=1):
        content_snippet = doc.content[:1200]
        if len(doc.content) > 1200:
            content_snippet += "..."

        blocks.append(
            f"=== Support Document {i} ===\n"
            f"Title: {doc.title}\n"
            f"URL: {doc.url}\n"
            f"Content: {content_snippet}"
        )

        if doc.url and doc.url not in source_urls:
            source_urls.append(doc.url)

    return "\n\n".join(blocks), source_urls


def _build_system_prompt(domain: str) -> str:
    """Build the system instruction that constrains the model to the corpus."""
    domain_label = domain.title() if domain != "unknown" else "our product"
    return (
        f"You are a helpful, professional support agent for {domain_label}.\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using the support documents provided below as context.\n"
        "2. If the answer is not in the provided documents, say exactly: "
        f'"{_CORPUS_NOT_FOUND}"\n'
        "3. Do NOT use your general knowledge or training data.\n"
        "4. Do NOT make up policies, procedures, phone numbers, or features.\n"
        "5. Be concise, professional, and empathetic.\n"
        "6. When referencing information, mention which document it came from.\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def _generate_reply_with_retry(
    ticket_text: str,
    classification: Classification,
    retrieved_docs: list[Document],
) -> str:
    client = rotator.get_client()
    if not client:
        return _CORPUS_NOT_FOUND

    context, _ = _build_context(retrieved_docs)
    system_prompt = _build_system_prompt(classification.domain)
    user_prompt = (
        f"{system_prompt}\n\n"
        f"Customer ticket:\n{ticket_text}\n\n"
        f"Support documents:\n{context}\n\n"
        "Provide a helpful, concise response based strictly on the above documents."
    )

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
        )
    )
    return response.text.strip()


def generate_reply(
    ticket_text: str,
    classification: Classification,
    retrieved_docs: list[Document],
    index: dict,
) -> AgentResponse:
    """Generate a corpus-grounded reply using rotating Gemini API keys."""
    if not rotator.has_keys():
        return AgentResponse(action="reply", response=_CORPUS_NOT_FOUND, sources=[])

    # Stage 1: Identify domain and perform Targeted Search
    # If domain is 'unknown', we go straight to Global Search
    target_domain = classification.domain if classification.domain != "unknown" else None
    
    # Use higher top_k for targeted search to capture more context from the relevant area
    retrieved_docs = loader.search(
        ticket_text, 
        index, 
        domain=target_domain, 
        top_k=7 if target_domain else 5
    )

    # Stage 2: Global Fallback
    # If targeted search returned poor results (low top score) or no results, try global
    if target_domain and (not retrieved_docs or _get_top_score(ticket_text, retrieved_docs) < 10.0):
        print(f"[responder] Targeted search in '{target_domain}' was poor. Falling back to Global Search.")
        global_docs = loader.search(ticket_text, index, domain=None, top_k=5)
        # Merge results, prioritizing targeted docs but allowing global ones if they are much better
        retrieved_docs = global_docs
    
    if not retrieved_docs:
        return AgentResponse(action="reply", response=_CORPUS_NOT_FOUND, sources=[])

    _, source_urls = _build_context(retrieved_docs)

    try:
        reply_text = _generate_reply_with_retry(ticket_text, classification, retrieved_docs)
        
        # Internal check for empty/nonsensical replies
        if len(reply_text) < 10:
            return AgentResponse(action="reply", response=_CORPUS_NOT_FOUND, sources=source_urls)

        flagged = _check_hallucination(reply_text, retrieved_docs)
        if flagged:
            print(f"[responder] HALLUCINATION/PII WARNING: {len(flagged)} violation(s) flagged.")
            # If PII leak is detected, we fallback to a safe message
            if any("Leak" in f for f in flagged):
                return AgentResponse(action="reply", response=_CORPUS_NOT_FOUND, sources=source_urls)

        return AgentResponse(action="reply", response=reply_text, sources=source_urls)

    except Exception as exc:  # noqa: BLE001
        print(f"[responder] ERROR after retries: {exc}")
        return AgentResponse(action="reply", response=_CORPUS_NOT_FOUND, sources=source_urls)


def generate_escalation(
    ticket_text: str,  # noqa: ARG001
    safety_decision: SafetyDecision,
) -> AgentResponse:
    """Build a professional escalation notice WITHOUT calling the API."""
    message = (
        "Thank you for reaching out. Your request has been escalated to our "
        f"specialized support team because: {safety_decision.reason}. "
        "A human agent will contact you shortly."
    )
    return AgentResponse(action="escalate", response=message, sources=[])
