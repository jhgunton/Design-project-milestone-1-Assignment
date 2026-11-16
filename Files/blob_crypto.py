"""Fernet encryption for the answer-key blob (``.pset_checks.dat``).

The blob holds the autograded reference values / instructor code the local checker needs.
It is encrypted so the answers are not sitting in plaintext on the student's disk.

This is a speedbump, not a security boundary: the key below ships inside the student
checker, so a determined student can recover it and decrypt the blob. It stops casual
answer-peeking, which is the stated goal.

Imported only by the build and by ``pset_check`` -- never by the Gradescope grader, so
``cryptography`` is never required in the Gradescope container.
"""
from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet

# Embedded key, derived from a fixed passphrase so it is a valid 32-byte Fernet key
# without hand-encoding base64. Obfuscation-grade by design (it ships client-side).
_FERNET_KEY = base64.urlsafe_b64encode(
    hashlib.sha256(b"20.320 pset check blob key v1").digest()
)
_fernet = Fernet(_FERNET_KEY)


def encrypt(data: bytes) -> bytes:
    """Encrypt ``data`` (bytes) into a Fernet token."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("encrypt expects bytes")
    return _fernet.encrypt(bytes(data))


def decrypt(token: bytes) -> bytes:
    """Decrypt a Fernet token back to bytes.

    Raises ``cryptography.fernet.InvalidToken`` if the blob was tampered with or the key
    does not match -- fail loud, no silent fallback.
    """
    if isinstance(token, str):
        token = token.encode("utf-8")
    return _fernet.decrypt(token)
