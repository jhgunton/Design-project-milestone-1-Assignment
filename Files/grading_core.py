"""Shared grading core.

This module holds the comparison and block-execution logic that used to live inside
the Gradescope ``grade.py`` template. It is factored out so that exactly the same code
runs in two off-Gradescope places:

  * the build-time "precompute" pass (snapshots instructor reference values), and
  * the student-side ``pset_check.check_answer`` (in-notebook checking).

Keeping a single implementation guarantees that what a student sees locally matches what
the reference values encode -- there is no second grader to drift from.

NOTE: Gradescope itself does NOT import this module. Its thin grader only verifies a
signature and maps recorded pass/fail onto a baked-in point schema.
"""
from __future__ import annotations

import os

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is always present in practice
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is a project dependency
    plt = None

# Sentinel value to detect when an expected value is not provided
_NOT_PROVIDED = object()

# Variables that are infrastructure, not instructor answers, and must not be snapshotted
_SKIP_VARS = {
    "np",
    "plt",
    "__name__",
    "__builtins__",
    "compare_answers",
    "interactive_demo_launcher",
    "launch_standalone_interactive_demo",
}


def summarize_value(value):
    """Format a value for compact output (truncate large arrays / strings)."""
    if hasattr(value, "shape"):
        if value.size <= 4:
            return f"{value.flatten().tolist()}"
        return f"array{value.shape}"
    elif isinstance(value, str):
        if len(value) > 30:
            return f"'{value[:30]}...'"
        return repr(value)
    elif isinstance(value, float):
        return f"{value:.4g}"
    return repr(value)


def compare_answers(namespace, variable_name, expected=_NOT_PROVIDED, atol=None, rtol=None):
    """Compare a student answer to the instructor answer, auto-detecting the type.

    Args:
        namespace: The grading namespace holding both the student's ``variable_name``
            and the ``instructor_{variable_name}`` reference value.
        variable_name: Name of the variable to compare.
        expected: Explicit expected value. If omitted, ``instructor_{variable_name}``
            is looked up in ``namespace`` instead.
        atol: Absolute tolerance for numeric comparisons (default exact).
        rtol: Relative tolerance for numeric comparisons (overrides atol if set).

    Raises:
        AssertionError: if the student answer does not match.
    """
    if variable_name not in namespace:
        raise AssertionError(f"{variable_name} is not defined")
    student = namespace[variable_name]
    if expected is _NOT_PROVIDED:
        instructor = namespace[f"instructor_{variable_name}"]
    else:
        instructor = expected

    passed = True
    error_message = ""

    # String / choice comparison (case-insensitive)
    if isinstance(instructor, str):
        if student.upper() != instructor.upper():
            passed = False
            error_message = f"{variable_name}: incorrect"

    # Set comparison (case-insensitive for string elements)
    elif isinstance(instructor, set):
        def normalize_set(values):
            return {x.lower() if isinstance(x, str) else x for x in values}

        student_normalized = normalize_set(student)
        instructor_normalized = normalize_set(instructor)
        if student_normalized != instructor_normalized:
            passed = False
            missing = instructor_normalized - student_normalized
            extra = student_normalized - instructor_normalized
            error_message = f"{variable_name}:"
            if missing:
                error_message += f" missing {missing}"
            if extra:
                error_message += f" extra {extra}"

    # Numpy array comparison
    elif hasattr(instructor, "shape"):
        if rtol is not None:
            passed = np.allclose(student, instructor, rtol=rtol, atol=0)
        elif atol is not None:
            passed = np.allclose(student, instructor, atol=atol, rtol=0)
        else:
            passed = np.array_equal(student, instructor)
        if not passed:
            error_message = f"{variable_name}: arrays don't match"

    # Numeric comparison (int, float)
    elif isinstance(instructor, (int, float)):
        if rtol is not None:
            passed = abs(student - instructor) <= rtol * abs(instructor)
        elif atol is not None:
            passed = abs(student - instructor) <= atol
        else:
            passed = student == instructor
        if not passed:
            error_message = f"{variable_name}: incorrect"

    # Fallback: exact equality
    else:
        passed = student == instructor
        if not passed:
            error_message = f"{variable_name}: incorrect"

    if not passed:
        raise AssertionError(error_message)


def build_instructor_namespace(
    instructor_code,
    *,
    cwd=None,
    strip_files_prefix=False,
    stub_demos=True,
):
    """Execute instructor code and return the resulting namespace.

    The namespace is pre-seeded with the standard scientific-Python environment students have --
    ``np`` (numpy) and ``plt`` (matplotlib.pyplot) -- so answer code that also plots runs fine.
    Plotting is harmless here: ``plt.show`` is suppressed for the duration and any figures the code
    creates are closed afterwards, and the caller's own ``plt.show`` is restored (so running a check
    never disturbs a student's in-notebook plots). ``display`` is stubbed to a no-op for the same
    reason -- it is injected by IPython rather than imported, so it exists in a notebook but not in
    this plain exec(). Anything more exotic (helper modules, data loads) is declared by marking that
    cell ``@@APPEARS_IN precompute`` in the notebook.

    Still absent, and deliberately: ``get_ipython``, the magics, and the rest of
    ``IPython.display``. Instructor code needing those should carry
    ``@@APPEARS_IN ALL EXCEPT precompute`` rather than pull a kernel in here.

    The returned namespace contains both the instructor's plain variable names and an
    ``instructor_`` prefixed duplicate of each (the reference-value snapshot).

    The *environment seam* is parameterized because it differs by caller:
      * build precompute pass -> ``cwd=<instructor notebook dir>`` so data files resolve,
      * ``check_answer`` local mode -> ``cwd=None`` (the notebook's own cwd already works).

    Args:
        instructor_code: Concatenated GRADER/ANSWER/INCLUDE code (see
            ``autograder.extract_instructor_code_from_nb``).
        cwd: Directory to run in while executing (restored afterwards). ``None`` keeps
            the current working directory.
        strip_files_prefix: If True, strip a leading ``Files/`` from path literals in the
            instructor code before executing (Gradescope layout). Off for local use.
        stub_demos: If True, install no-op stubs for interactive demo helpers that would
            otherwise open ipywidgets.
    """
    namespace = {"__name__": "__main__"}
    if np is not None:
        namespace["np"] = np
    # Provide the same plotting library students have, so answer code that also plots runs fine.
    if plt is not None:
        namespace["plt"] = plt

    def _noop(*args, **kwargs):
        return None

    # ``display`` is not a builtin -- a notebook has it only because IPython injects it into the
    # interactive namespace, and this is a plain exec(). A cell ending in display(some_table) then
    # works in VS Code and fails only here, which reads as a build bug rather than an authoring
    # one. Stub it for the same reason plt.show is suppressed below: its only effect is rendering,
    # and precompute exists to compute reference values, not to draw. This also matters beyond the
    # build -- in CHECK local mode the instructor code re-runs through this namespace inside the
    # student's own notebook.
    namespace["display"] = _noop

    # @@AUTOGRADED bodies run here too, so that a block can derive the value it checks (see
    # _PRESET_DESTINATIONS in cell_transform.py). Their compare_answers() calls cannot do anything
    # at build time -- there is no student answer yet, only the instructor's. Rather than ignore
    # them, use the call to verify that the block actually produced the name it claims to check.
    # A typo there would otherwise ship a template whose checks can never pass, reported to the
    # student as "0/N checks passed" with no hint that the fault is the assignment's.
    def _compare_answers_at_build_time(variable_name, *args, **kwargs):
        if variable_name not in namespace:
            raise NameError(
                f'compare_answers("{variable_name}") names a variable that does not exist after '
                "the instructor code ran. Define it in the @@AUTOGRADED block or in a cell that "
                "reaches precompute, or correct the name."
            )
        return None

    namespace["compare_answers"] = _compare_answers_at_build_time

    if stub_demos:
        namespace["interactive_demo_launcher"] = _noop
        namespace["launch_standalone_interactive_demo"] = _noop

    code = instructor_code
    if strip_files_prefix:
        code = code.replace("'Files/", "'").replace('"Files/', '"')

    # Run answer code that plots without displaying anything or leaking figures: suppress show()
    # for the duration and close only the figures this exec creates, then restore the student's
    # own plt.show (so their in-notebook plotting is untouched by a check).
    figures_before = set(plt.get_fignums()) if plt is not None else set()
    saved_show = plt.show if plt is not None else None
    if plt is not None:
        plt.show = _noop

    old_cwd = os.getcwd()
    try:
        if cwd is not None:
            os.chdir(str(cwd))
        exec(code, namespace)
    finally:
        if cwd is not None:
            os.chdir(old_cwd)
        if plt is not None:
            plt.show = saved_show
            for number in set(plt.get_fignums()) - figures_before:
                plt.close(number)

    # Duplicate every instructor-defined variable under an instructor_ prefix, preserving
    # the reference values before any student code can overwrite the plain names.
    for key, value in list(namespace.items()):
        if key.startswith("_") or key in _SKIP_VARS:
            continue
        namespace[f"instructor_{key}"] = value

    return namespace


def instructor_snapshot(namespace):
    """Return only the ``instructor_`` prefixed values from a namespace.

    Used by the build precompute pass to pickle a minimal reference-value snapshot.
    """
    return {key: value for key, value in namespace.items() if key.startswith("instructor_")}


def run_block(namespace, code):
    """Execute one AUTOGRADED test block in ``namespace``.

    A fresh ``compare_answers`` bound to ``namespace`` is injected so instructor test
    code can call ``compare_answers("var", atol=...)`` with one argument.

    Returns:
        ``(passed, message)`` -- ``passed`` is False if the block raised, with the
        exception text in ``message``.
    """
    namespace["compare_answers"] = lambda variable_name, expected=_NOT_PROVIDED, atol=None, rtol=None: compare_answers(
        namespace, variable_name, expected, atol, rtol
    )
    try:
        exec(code, namespace)
        return True, ""
    except Exception as error:  # noqa: BLE001 - any failure means the test did not pass
        return False, str(error)
