"""Student-facing in-notebook answer checker.

Shipped into the student template's ``Files/`` folder. Provides:

  * ``check_answer("1.2")`` -- grade one auto-graded question against the encrypted
    reference values, print pass/fail, and record the result to a local log.
  * ``build_signed_results(student_email, ...)`` -- read the log, enforce that every auto
    question was checked, and produce an identity-bound signed payload for submission.

The reference answers live in an encrypted blob (``.pset_checks.dat``) next to this file.
This is a speedbump, not airtight security: the decryption key ships in ``blob_crypto``.
Grade integrity ultimately rests on the PDF audit trail, not on this checker.
"""
from __future__ import annotations

import inspect
import json
import pickle
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import blob_crypto
    import grading_core
    import signing
except ImportError:
    from . import blob_crypto, grading_core, signing  # type: ignore

_HERE = Path(__file__).resolve().parent
_BLOB_PATH = _HERE / ".pset_checks.dat"
_LOG_PATH = _HERE / ".pset_check_log.json"

# Cache the decrypted blob so repeated check_answer calls don't re-decrypt.
_blob_cache = None
# Cache the instructor reference values built from local-mode instructor code.
_local_instructor_values = None


def _load_blob():
    """Decrypt and cache the answer-key blob. Fail loud if it is missing/tampered."""
    global _blob_cache
    if _blob_cache is None:
        if not _BLOB_PATH.exists():
            raise FileNotFoundError(
                f"Answer-check data not found at {_BLOB_PATH}. Make sure you are running "
                "inside the released student notebook with its Files/ folder intact."
            )
        _blob_cache = pickle.loads(blob_crypto.decrypt(_BLOB_PATH.read_bytes()))
    return _blob_cache


def _local_values():
    """Build (and cache) instructor reference values for CHECK local questions."""
    global _local_instructor_values
    if _local_instructor_values is None:
        blob = _load_blob()
        instructor_code = blob.get("instructor_code", "")
        namespace = grading_core.build_instructor_namespace(
            instructor_code, cwd=None, strip_files_prefix=False
        )
        _local_instructor_values = grading_core.instructor_snapshot(namespace)
    return _local_instructor_values


def _load_log():
    if _LOG_PATH.exists():
        try:
            return json.loads(_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_log(log):
    _LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")


def check_answer(question_id, namespace=None):
    """Grade one auto-graded question and record the result.

    Args:
        question_id: The question label, e.g. ``"1.2"`` (matches the notebook header).
        namespace: Variable namespace to grade. Defaults to the caller's globals, so in
            a notebook you simply call ``check_answer("1.2")`` after your solution cell.

    Returns:
        True if every check for the question passed, else False.
    """
    question_id = str(question_id)
    if namespace is None:
        namespace = inspect.stack()[1].frame.f_globals

    blob = _load_blob()
    blocks = [b for b in blob["blocks"] if str(b["display"]) == question_id]
    if not blocks:
        raise ValueError(
            f'No auto-graded checks found for question "{question_id}". '
            "Check the question number in the notebook header."
        )

    # Assemble the reference (instructor_*) values these blocks need.
    instructor_values = dict(blob.get("precomputed", {}))
    if any(b.get("mode") == "local" for b in blocks):
        instructor_values.update(_local_values())

    # Build the grading namespace: numpy, instructor_ reference values, then the student's
    # own variables on top (so plain names resolve to the student's work).
    run_namespace = {"__name__": "__grading__"}
    if np is not None:
        run_namespace["np"] = np
    run_namespace.update(instructor_values)
    for key, value in namespace.items():
        if key.startswith("__"):
            continue
        run_namespace[key] = value

    results = []
    for block in blocks:
        passed, message = grading_core.run_block(run_namespace, block["code"])
        results.append((block, passed, message))

    _print_results(question_id, results)

    # Record pass/fail per block for this question (overwriting any prior run).
    log = _load_log()
    log[question_id] = {block["name"]: bool(passed) for block, passed, _ in results}
    _save_log(log)

    return all(passed for _, passed, _ in results)


def _print_results(question_id, results):
    """Print pass/fail for a question. Never reveals expected values."""
    def undefined_hint(message):
        if "is not defined" in message:
            return "  (did you run your solution cell above first?)"
        return ""

    if len(results) == 1:
        block, passed, message = results[0]
        if passed:
            print(f"✅ Question {question_id}: correct")
        else:
            print(f"❌ Question {question_id}: not correct yet")
            note = undefined_hint(message)
            if note:
                print(note)
            elif block.get("hint") and message:
                print(f"  hint: {message}")
        return

    all_passed = all(passed for _, passed, _ in results)
    header = "✅" if all_passed else "❌"
    print(f"{header} Question {question_id}: "
          f"{sum(1 for _, p, _ in results if p)}/{len(results)} checks passed")
    for block, passed, message in results:
        mark = "✅" if passed else "❌"
        print(f"  {mark} {block['name']}")
        if not passed:
            note = undefined_hint(message)
            if note:
                print(f"  {note.strip()}")
            elif block.get("hint") and message:
                print(f"      hint: {message}")


def build_signed_results(student_email, allow_unchecked=False):
    """Build the signed results payload for submission.

    Reads the recorded log and requires that every auto-graded question was checked.

    Args:
        student_email: Email address associated with the student's Gradescope account.
            It is normalized and included inside the signed payload.
        allow_unchecked: If False (default) and any question was never checked, raise
            loudly. If True, submit partial work: unchecked questions are recorded as
            failed (no credit).

    Returns:
        ``{"payload": {identity, assignment, results}, "signature": <hex>}``
    """
    normalized_email = signing.normalize_email(student_email)
    blob = _load_blob()
    blocks = blob["blocks"]
    assignment_id = blob.get("assignment_id")
    if not isinstance(assignment_id, str) or not assignment_id:
        raise RuntimeError(
            "This assignment's answer-check data predates identity-bound submissions. "
            "Download the current assignment template before submitting."
        )
    log = _load_log()

    expected_questions = sorted({str(b["display"]) for b in blocks})
    unchecked = [q for q in expected_questions if q not in log]
    if unchecked and not allow_unchecked:
        raise RuntimeError(
            "These auto-graded questions have not been checked yet: "
            f"{', '.join(unchecked)}.\n"
            "Run check_answer(\"<question>\") for each, or re-run with "
            "allow_unchecked=True to submit partial work (unchecked questions score 0)."
        )

    results = []
    for block in blocks:
        display = str(block["display"])
        passed = bool(log.get(display, {}).get(block["name"], False))
        results.append({"name": block["name"], "display": display, "passed": passed})

    payload = {
        "format_version": signing.SIGNED_RESULTS_FORMAT_VERSION,
        "assignment_id": assignment_id,
        "student_email": normalized_email,
        "results": results,
    }
    return {"payload": payload, "signature": signing.sign(payload)}
