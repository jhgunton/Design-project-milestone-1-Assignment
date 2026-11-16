"""Detached HMAC signing for the student results file (stdlib only).

The student's identity, assignment id, and ``check_answer`` results are signed here at
submission time; the thin Gradescope grader verifies the signature before trusting the
recorded pass/fail. Only integrity is needed (the scores are not secret), so a stdlib HMAC
is sufficient -- and being dependency-free means the Gradescope container needs nothing
beyond the standard library to verify.

This is a speedbump, not a security boundary: the key below ships inside the student
checker, so a determined student can forge a signature. Grade integrity ultimately rests
on the PDF audit trail, not on this signature.
"""
from __future__ import annotations

import hashlib
import hmac
import json

# Embedded key. Obfuscation-grade by design (it ships client-side). Derived from a fixed
# passphrase so it is a stable constant without hand-encoding bytes.
_SIGNING_KEY = hashlib.sha256(b"20.320 pset signed-results key v1").digest()

SIGNED_RESULTS_FORMAT_VERSION = 2
STUDENT_EMAIL_PLACEHOLDER = "your_email@mit.edu"


def normalize_email(value) -> str:
    """Return the canonical form used when signing and comparing student email addresses.

    Deliberately performs only structural validation. Gradescope accounts are not restricted to
    MIT addresses, and attempting to implement the full email-address grammar here would reject
    valid accounts without adding useful integrity.
    """
    if not isinstance(value, str):
        raise ValueError("my_email must be a string.")

    email = value.strip().casefold()
    if not email or email == STUDENT_EMAIL_PLACEHOLDER:
        raise ValueError(
            "Set my_email to the email address associated with your Gradescope account."
        )
    if email.count("@") != 1 or not all(email.split("@", 1)):
        raise ValueError("my_email must be a valid email address, such as name@mit.edu.")
    return email


def canonical_bytes(payload) -> bytes:
    """Canonical JSON encoding used on both the signing and verifying sides.

    Both sides MUST encode identically or verification fails, so this is the single
    source of truth for the byte representation that gets signed.
    """
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign(payload) -> str:
    """Return the hex HMAC-SHA256 signature of ``payload`` (bytes or JSON-able object)."""
    return hmac.new(_SIGNING_KEY, canonical_bytes(payload), hashlib.sha256).hexdigest()


def verify(payload, signature) -> bool:
    """Return True iff ``signature`` matches ``payload`` under the embedded key."""
    if not isinstance(signature, str):
        return False
    return hmac.compare_digest(sign(payload), signature)
