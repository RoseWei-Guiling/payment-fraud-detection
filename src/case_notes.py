"""
Step 8: case notes for fraud investigators.

When a payment is blocked, an investigator needs a short note: what happened, why it was
stopped, what to check. An LLM can write that note well, but it must not invent facts.
So:
  1. We build the facts ourselves (amount, customer behaviour, score, reasons).
  2. The LLM may only rewrite those facts into a note (a strict instruction).
  3. Every number in the note is checked against the facts. If the LLM wrote a number that
     isn't there, the note is rejected and a plain template note is used instead.

Without an API key, the template is used for every note, so the step works offline.
This dataset is anonymised and public. With real customer data, sending it to an external
LLM would need data-protection approval first.
"""
import re

import numpy as np
import pandas as pd

SYSTEM = (
    "You write case notes for card fraud investigators at a bank. Use ONLY the facts given. "
    "Do not add any number, name or event that is not in the facts. Write 3 short sentences: "
    "what the payment was, why the model stopped it, and the most useful thing to check with the "
    "customer. Plain English, no bullet points, no speculation about who the customer is."
)


def _value(row, col, fmt=None):
    v = row.get(col)
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return fmt(v) if fmt else v


def case_facts(row, result):
    """Facts for one payment, from its features (row: dict) and its API result."""
    facts = {
        "payment_id": _value(row, "TransactionID", int),
        "amount": _value(row, "TransactionAmt", lambda v: round(float(v), 2)),
        "product_type": _value(row, "ProductCD", str),
        "card_network": _value(row, "card4", str),
        "card_type": _value(row, "card6", str),
        "purchaser_email_domain": _value(row, "P_emaildomain", str),
        "device": _value(row, "DeviceType", str),
        "customer_payments_last_24h": _value(row, "uid_n_24h", lambda v: int(v)),
        "customer_payment_number": _value(row, "uid_txn_number", lambda v: int(v)),
        "first_time_on_this_device": _value(row, "new_fingerprint_for_uid", lambda v: "yes" if v else "no"),
        "probability_of_fraud_pct": round(100 * result["probability_of_fraud"], 1),
        "decision": result["decision"],
        "model_reasons": result["reasons"],
        "scorecard_reasons": result.get("scorecard_reasons", []),
    }
    return {k: v for k, v in facts.items() if v is not None and v != []}


def facts_text(facts):
    lines = []
    for k, v in facts.items():
        lines.append(f"- {k}: {'; '.join(v) if isinstance(v, list) else v}")
    return "\n".join(lines)


def template_note(facts):
    who = (f"the customer's payment number {facts['customer_payment_number']}"
           if "customer_payment_number" in facts else "a payment")
    reasons = "; ".join(facts.get("scorecard_reasons") or facts.get("model_reasons", []))
    return (f"Payment {facts.get('payment_id', '')} of {facts.get('amount')} was {facts['decision']}ed "
            f"with a {facts['probability_of_fraud_pct']}% fraud probability; it was {who}. "
            f"Main reasons: {reasons}. "
            "Check with the customer that they made this payment and recognise the device used.")


def numbers_in(text):
    return {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def is_grounded(note, facts):
    """True if every number in the note also appears in the facts."""
    allowed = numbers_in(facts_text(facts))
    return all(any(np.isclose(n, a) for a in allowed) for n in numbers_in(note))


def llm_note(facts, client, model):
    message = client.messages.create(
        model=model, max_tokens=300, system=SYSTEM,
        messages=[{"role": "user", "content": "Facts:\n" + facts_text(facts)}])
    return message.content[0].text.strip()


def write_note(facts, client=None, model=None):
    """Returns (note, source). Falls back to the template if no LLM or if the LLM's note
    contains a number that is not in the facts."""
    if client is None:
        return template_note(facts), "template"
    try:
        note = llm_note(facts, client, model)
    except Exception as e:  # network or API error: never lose the note
        return template_note(facts), f"template (LLM error: {type(e).__name__})"
    if not is_grounded(note, facts):
        return template_note(facts), "template (LLM note rejected: number not in the facts)"
    return note, f"LLM ({model})"
