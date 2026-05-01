"""
main.py — Terminal entry point for the support triage agent.

Ties together all modules: corpus loader → classifier → retriever →
safety gate → responder → logger, writing results to an output CSV.

Usage (all args have defaults so bare `python main.py` works):
    python main.py
    python main.py --input support_tickets/support_tickets.csv
    python main.py --dry-run                  # classify + retrieve only, no Gemini reply
    python main.py --ticket-id T003           # process a single ticket
    python main.py --input support_tickets/sample_support_tickets.csv \\
                   --output support_tickets/output.csv \\
                   --data data/ \\
                   --log log.txt
"""

import argparse
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path setup — works regardless of where python is invoked from
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "code"))

import pandas as pd

from agent.classifier import Classification, classify
from agent.responder import AgentResponse, generate_escalation, generate_reply
from agent.safety import SafetyDecision, check as safety_check
from corpus.loader import build_index, load_corpus, search
from utils.logger import TriageLogger, print_progress

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT  = _REPO_ROOT / "support_tickets" / "support_tickets.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "support_tickets" / "output.csv"
DEFAULT_DATA   = _REPO_ROOT / "data"
DEFAULT_LOG    = _REPO_ROOT / "log.txt"

RETRIEVAL_TOP_K = 5

# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

# Maps actual CSV column names → canonical internal names.
# The spec uses ticket_id / ticket_text; the repo CSV uses Issue / Subject / Company.
_COLUMN_ALIASES: dict[str, list[str]] = {
    "ticket_id":   ["ticket_id", "id", "ID"],
    "ticket_text": ["ticket_text", "Issue", "issue", "text"],
    "subject":     ["subject", "Subject", "SUBJECT"],
    "company":     ["company", "Company", "COMPANY"],
    "email":       ["customer_email", "email", "Email"],
    "created_at":  ["created_at", "CreatedAt", "date"],
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename actual CSV columns to canonical internal names.

    Walks _COLUMN_ALIASES; if any alias is present in the DataFrame it is
    renamed to the canonical name. Columns not in the alias table are kept
    unchanged. A synthetic `ticket_id` column (T001, T002 …) is added when
    none is found.

    Args:
        df: Raw DataFrame as read from the input CSV.

    Returns:
        DataFrame with canonical column names where possible.
    """
    rename_map: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and alias != canonical:
                rename_map[alias] = canonical
                break

    df = df.rename(columns=rename_map)

    # Synthesise ticket_id if not present
    if "ticket_id" not in df.columns:
        df.insert(0, "ticket_id", [f"T{i:03d}" for i in range(1, len(df) + 1)])

    # Merge subject into ticket_text when both exist
    if "ticket_text" in df.columns and "subject" in df.columns:
        df["ticket_text"] = df.apply(
            lambda r: (
                f"{r['subject']}\n{r['ticket_text']}"
                if pd.notna(r.get("subject")) and str(r.get("subject", "")).strip()
                   not in ("", "None", "none")
                else str(r["ticket_text"])
            ),
            axis=1,
        )
    elif "ticket_text" not in df.columns:
        raise ValueError(
            "Input CSV must have at least an 'Issue' or 'ticket_text' column."
        )

    return df


# ---------------------------------------------------------------------------
# request_type → evaluator-expected label mapping
# ---------------------------------------------------------------------------

_RT_MAP: dict[str, str] = {
    "bug":             "bug",
    "feature_request": "feature_request",
    "billing":         "product_issue",
    "account_access":  "product_issue",
    "fraud":           "product_issue",
    "permissions":     "product_issue",
    "assessment":      "product_issue",
    "faq":             "product_issue",
    "other":           "invalid",
}

# Tickets that are out-of-scope / nonsensical map to "invalid"
_INVALID_DOMAINS = {"unknown"}


def _map_request_type(rt: str, domain: str) -> str:
    """Map internal classifier request_type to the allowed output label.

    Args:
        rt:     Internal request_type string from Classification.
        domain: Classified domain; 'unknown' forces 'invalid'.

    Returns:
        One of: product_issue | feature_request | bug | invalid
    """
    if domain in _INVALID_DOMAINS and rt == "other":
        return "invalid"
    return _RT_MAP.get(rt, "product_issue")


# ---------------------------------------------------------------------------
# Per-ticket pipeline
# ---------------------------------------------------------------------------


def process_ticket(
    ticket_id: str,
    ticket_text: str,
    company: str,
    index: dict,
    dry_run: bool = False,
) -> dict:
    """Run the full triage pipeline for one ticket.

    Steps:
        a. classify()       → Classification
        b. search()         → retrieved_docs (domain-scoped, global fallback)
        c. safety_check()   → SafetyDecision
        d. generate_*()     → AgentResponse   (skipped in dry_run mode)
        e. print_progress() → terminal line

    Args:
        ticket_id:   Identifier string (e.g. "T001").
        ticket_text: Combined subject + issue text.
        company:     Raw Company field value from CSV (may be empty).
        index:       Pre-built BM25 index dict.
        dry_run:     If True, skip the Gemini response call and return a
                     stub reply. Useful for testing classify + retrieve
                     without consuming API quota.

    Returns:
        Dict with keys matching the output CSV schema.
    """
    from corpus.loader import infer_domain_from_search

    # a. Classify
    classification = classify(ticket_text)

    # Override domain from CSV Company column when classifier returns unknown
    _COMPANY_DOMAIN = {"hackerrank": "hackerrank", "claude": "claude", "visa": "visa"}
    if classification.domain == "unknown" and company:
        inferred = _COMPANY_DOMAIN.get(company.strip().lower())
        if inferred:
            # Boost confidence: company column is authoritative ground truth
            new_conf = max(classification.confidence, 0.70)
            classification = Classification(
                domain=inferred,
                request_type=classification.request_type,
                product_area=classification.product_area,
                confidence=new_conf,
            )
    elif classification.domain != "unknown" and company:
        # Even when domain matches, boost low confidence from company signal
        inferred = _COMPANY_DOMAIN.get(company.strip().lower())
        if inferred and inferred == classification.domain and classification.confidence < 0.60:
            classification = Classification(
                domain=classification.domain,
                request_type=classification.request_type,
                product_area=classification.product_area,
                confidence=max(classification.confidence, 0.60),
            )

    # If domain is still unknown, infer from cross-domain BM25 search
    if classification.domain == "unknown":
        inferred_domain = infer_domain_from_search(ticket_text, index)
        if inferred_domain:
            classification = Classification(
                domain=inferred_domain,
                request_type=classification.request_type,
                product_area=classification.product_area,
                confidence=max(classification.confidence, 0.55),
            )

    # b. Retrieve — domain-scoped first, then global fallback
    domain = classification.domain if classification.domain != "unknown" else None
    retrieved = search(ticket_text, index, domain=domain, top_k=RETRIEVAL_TOP_K, min_score=1.0)
    if not retrieved and domain:
        retrieved = search(ticket_text, index, domain=None, top_k=RETRIEVAL_TOP_K, min_score=1.0)

    # c. Safety gate
    decision = safety_check(ticket_text, classification, retrieved)

    # d. Generate response (skipped in dry-run mode)
    if dry_run:
        action   = "dry-run"
        response_text = (
            f"[DRY RUN] domain={classification.domain} "
            f"request_type={classification.request_type} "
            f"retrieved={len(retrieved)} docs "
            f"would_escalate={decision.should_escalate}"
        )
        sources: list[str] = []
    elif decision.should_escalate:
        resp    = generate_escalation(ticket_text, decision)
        action  = resp.action
        response_text = resp.response
        sources = resp.sources
    else:
        resp    = generate_reply(ticket_text, classification, retrieved, index)
        action  = resp.action
        response_text = resp.response
        sources = resp.sources

    # e. Terminal progress
    print_progress(ticket_id, action, classification.domain)

    # Map action to evaluator expected status ('replied' or 'escalated')
    if action == "reply":
        status = "replied"
        # Build a concise prose justification explaining the routing decision.
        # Evaluators expect a human-readable explanation, not a raw URL list.
        domain_label = classification.domain.title() if classification.domain != "unknown" else "the relevant product"
        area_label   = classification.product_area or "general support"
        n_docs       = len(retrieved)
        justification = (
            f"Classified as a {classification.request_type.replace('_', ' ')} for "
            f"{domain_label} ({area_label}). "
            f"Response grounded in {n_docs} corpus document(s) retrieved via BM25 search. "
            f"No safety rules triggered; confidence score {classification.confidence:.2f}."
        )
    elif action == "escalate":
        status = "escalated"
        justification = decision.reason
    else:
        status = action
        justification = "Dry run mode — no LLM response generated"

    # Evaluators expect exactly these columns
    out = {
        "ticket_id":     ticket_id,
        "status":        status,
        "product_area":  f"{classification.domain} - {classification.product_area}",
        "response":      response_text,
        "justification": justification,
        "request_type":  _map_request_type(classification.request_type, classification.domain),
    }

    return {
        "out": out,
        "classification": classification,
        "safety_decision": decision,
        "retrieved_docs": retrieved,
        "agent_response": resp if not dry_run and action != "dry-run" else AgentResponse(action=action, response=response_text),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the full triage pipeline."""
    parser = argparse.ArgumentParser(
        description="Support Triage Agent — classifies and responds to support tickets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",     type=Path, default=DEFAULT_INPUT,
                        help="Input tickets CSV path")
    parser.add_argument("--output",    type=Path, default=DEFAULT_OUTPUT,
                        help="Output results CSV path")
    parser.add_argument("--data",      type=Path, default=DEFAULT_DATA,
                        help="Corpus data directory")
    parser.add_argument("--log",       type=Path, default=DEFAULT_LOG,
                        help="Log file path")
    parser.add_argument("--dry-run",   action="store_true", default=False,
                        help="Classify and retrieve only — skip Gemini response (saves API quota)")
    parser.add_argument("--ticket-id", type=str, default=None,
                        help="Process only the ticket with this ID (e.g. T003)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    dry_run_label = "  [DRY RUN — no Gemini response calls]" if args.dry_run else ""
    print(f"\n{'='*60}")
    print(f"  Support Triage Agent v1.0{dry_run_label}")
    print(f"{'='*60}")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Data   : {args.data}")
    print(f"  Log    : {args.log}")
    if args.ticket_id:
        print(f"  Filter : ticket_id = {args.ticket_id}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load + index corpus
    # ------------------------------------------------------------------
    print("Loading corpus...")
    docs = load_corpus(str(args.data))
    index = build_index(docs)
    print(f"Corpus loaded: {len(docs)} documents indexed\n")

    # ------------------------------------------------------------------
    # Read input CSV
    # ------------------------------------------------------------------
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input, dtype=str).fillna("")
    df = _normalise_columns(df)

    # Filter to single ticket when --ticket-id is set
    if args.ticket_id:
        mask = df["ticket_id"].str.upper() == args.ticket_id.upper()
        df = df[mask].reset_index(drop=True)
        if df.empty:
            print(f"ERROR: No ticket found with id '{args.ticket_id}'.")
            print(f"Available ids: {list(df['ticket_id'])}")
            sys.exit(1)
        print(f"Single-ticket mode: running only {args.ticket_id}\n")

    print(f"Processing {len(df)} ticket(s){'  [DRY RUN]' if args.dry_run else ''}...\n")

    # ------------------------------------------------------------------
    # Per-ticket pipeline (Parallel)
    # ------------------------------------------------------------------
    logger = TriageLogger(str(args.log))
    
    # Using ThreadPoolExecutor to process tickets in parallel
    # max_workers=7 because we have 7 rotating API keys
    MAX_WORKERS = 7 if not args.dry_run else 10
    
    tasks = []
    for _, row in df.iterrows():
        tasks.append({
            "ticket_id": str(row.get("ticket_id", f"T{len(tasks)+1:03d}")),
            "ticket_text": str(row.get("ticket_text", "")).strip(),
            "company": str(row.get("company", "")).strip()
        })

    results_dict = {} # Use dict to maintain order after parallel completion

    def _worker(task):
        t_id = task["ticket_id"]
        t_text = task["ticket_text"]
        t_comp = task["company"]
        try:
            res = process_ticket(t_id, t_text, t_comp, index, dry_run=args.dry_run)
            return t_id, res
        except Exception as exc:
            print(f"  [ERROR] {t_id}: {exc}")
            out = {
                "ticket_id":     t_id,
                "status":        "error",
                "product_area":  "unknown",
                "response":      f"Processing error: {exc}",
                "justification": traceback.format_exc(limit=2),
                "request_type":  "invalid",
            }
            res = {
                "out": out,
                "classification": Classification("unknown", "other", "unknown", 0.0),
                "safety_decision": SafetyDecision(True, f"Error: {exc}"),
                "retrieved_docs": [],
                "agent_response": AgentResponse("error", f"Error: {exc}"),
            }
            return t_id, res

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for future in as_completed(futures):
            t_id, res = future.result()
            results_dict[t_id] = res
            
            # Log immediately (logger is now thread-safe)
            logger.log_ticket(
                ticket_id=t_id,
                ticket_text=futures[future]["ticket_text"],
                classification=res["classification"],
                safety_decision=res["safety_decision"],
                retrieved_docs=res["retrieved_docs"],
                agent_response=res["agent_response"],
            )

    # Re-assemble results in original order
    results = [results_dict[task["ticket_id"]]["out"] for task in tasks]
    
    replied_n = sum(1 for r in results if r["status"] == "replied")
    escalated_n = sum(1 for r in results if r["status"] == "escalated")
    error_n = sum(1 for r in results if r["status"] == "error")

    logger.close()

    # ------------------------------------------------------------------
    # Write output CSV
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False, encoding="utf-8")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(results)
    print(f"\n{'='*60}")
    print("  === RESULTS ===")
    print(f"  Total: {total} | Replied: {replied_n} | "
          f"Escalated: {escalated_n} | Errors: {error_n}")
    print(f"  Output saved to {args.output}")
    print(f"  Log saved to {args.log}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
