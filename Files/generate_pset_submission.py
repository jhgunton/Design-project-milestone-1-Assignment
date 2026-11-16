#!/usr/bin/env python3
"""generate_pset_submission.py

Student-facing tool to generate Gradescope submission artifacts from a completed
assignment notebook.

Outputs (in <PSET>_to_submit/ subfolder, for example PSET1_to_submit/):
  1. signed_results.json — Identity-bound recorded auto scores (for autograding)
  2. code.py             — All code from auto-graded cells (submission record)
  3. answers_<email-id>.pdf — Rendered notebook with all questions (for manual review)

Usage (from within a notebook cell):
    from generate_pset_submission import generate_pset_submission
    generate_pset_submission('my_assignment.ipynb', student_email=my_email)

Or from command line:
    python psetTools/generate_pset_submission.py my_assignment.ipynb

Requirements:
    - nbformat
    - weasyprint (install via: conda install -c conda-forge weasyprint)
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nbformat
from nbformat import NotebookNode

try:
    from . import signing
    from .answer_pages import AnswerResponse, read_answer_response
except ImportError:
    # This module is copied into Files/ and imported as a top-level module in released notebooks.
    import signing
    from answer_pages import AnswerResponse, read_answer_response

try:
    from weasyprint import HTML as WeasyprintHTML
except ImportError:
    raise ImportError(
        "weasyprint is required for PDF generation.\n"
        "Install it with: conda install -c conda-forge weasyprint\n"
        "Or: pip install weasyprint"
    )


# -----------------------------
# Regex patterns (matching release_assignment.py conventions)
# -----------------------------

# Questions are found via the STYLED_* patterns below, which match the HTML the build emits.
# The pre-@@ formats (<!-- QUESTION ... --> and # QUESTION ...) were removed: the build has not
# emitted them for a long time, and matching them was dead code.

# Legacy format (transformed by older versions of the build tool):
# <span style="...">1.1 Question Title <span ...>(10 pts — manual)</span></span>
# or for autograded:
# <span style="...">1.1 Question Title <span ...>(10 pts — autograded)</span></span>
STYLED_SUBPART_RE = re.compile(
    r'<span[^>]*color:\s*#7030A0[^>]*>'  # Purple styled span
    r'\s*(\d+\.\d+)\s+'                   # Capture: "1.1 " (subpart number)
    r'([^<]+?)\s*'                        # Capture: title
    r'<span[^>]*>\s*'                     # Inner span for points
    r'\((\d+)\s+pts?\s*[—-]\s*(manual|autograded)\)'  # Capture: points and type
    r'\s*</span>\s*</span>'
)

# Problem header format (no points - just a container for subparts):
# <span style="...">Problem 2. Title</span>
STYLED_PROBLEM_RE = re.compile(
    r'<span[^>]*color:\s*#7030A0[^>]*font-size:\s*1\.5em[^>]*>'  # Purple, larger styled span
    r'\s*Problem\s+(\d+)\.\s+'             # Capture: problem number
    r'([^<]+?)\s*'                          # Capture: title
    r'</span>'
)

# Standalone-question header format (a whole question graded as one item, so it carries points):
# <span style="...">Problem 1. Key concepts <span ...>(11 pts — manual)</span></span>
# Its display number has no subpart dot, so STYLED_SUBPART_RE misses it.
STYLED_STANDALONE_RE = re.compile(
    r'<span[^>]*color:\s*#7030A0[^>]*>'
    r'\s*Problem\s+(\d+)\.\s+'                        # Capture: question number
    r'([^<]+?)\s*'                                    # Capture: title
    r'<span[^>]*>\s*'
    r'\((\d+)\s+pts?\s*[—-]\s*(manual|autograded)\)'  # Capture: points and type
    r'\s*</span>\s*</span>'
)

# Current format. Titles live in real Markdown headings so notebook frontends process inline math
# normally. The empty marker carries machine-readable identity without wrapping the title itself.
MARKDOWN_SUBPART_RE = re.compile(
    r'^###\s+<span\s+class="question-heading-marker"\s+'
    r'data-kind="subpart"\s+data-number="(\d+\.\d+)"></span>'
    r'\s*\1\s+(.+?)\s*'
    r'<span\s+class="question-heading-points">\s*'
    r'\((\d+)\s+pts?\s*[—-]\s*(manual|autograded)\)'
    r'\s*</span>\s*$',
    re.MULTILINE,
)

MARKDOWN_PROBLEM_RE = re.compile(
    r'^##\s+<span\s+class="question-heading-marker"\s+'
    r'data-kind="problem"\s+data-number="(\d+)"></span>'
    r'\s*Problem\s+\1\.\s+(.+?)\s*$',
    re.MULTILINE,
)

MARKDOWN_STANDALONE_RE = re.compile(
    r'^##\s+<span\s+class="question-heading-marker"\s+'
    r'data-kind="standalone"\s+data-number="(\d+)"></span>'
    r'\s*Problem\s+\1\.\s+(.+?)\s*'
    r'<span\s+class="question-heading-points">\s*'
    r'\((\d+)\s+pts?\s*[—-]\s*(manual|autograded)\)'
    r'\s*</span>\s*$',
    re.MULTILINE,
)


def _header_matches(source: str, current_pattern: re.Pattern,
                    legacy_pattern: re.Pattern):
    """Return current and legacy header matches in source order."""
    return iter(sorted(
        [*current_pattern.finditer(source), *legacy_pattern.finditer(source)],
        key=lambda match: match.start(),
    ))


def _subpart_header_matches(source: str):
    return _header_matches(source, MARKDOWN_SUBPART_RE, STYLED_SUBPART_RE)


def _standalone_header_matches(source: str):
    return _header_matches(source, MARKDOWN_STANDALONE_RE, STYLED_STANDALONE_RE)


def _problem_header_match(source: str):
    matches = list(_header_matches(source, MARKDOWN_PROBLEM_RE, STYLED_PROBLEM_RE))
    return matches[0] if matches else None

# Kept in sync BY HAND with notebook_types.AUTOG_BEGIN_RE / AUTOG_END_RE, and deliberately not
# imported: this module is copied standalone into the student's Files/ (release_assignment.py
# ships it there), where notebook_types.py does not exist. An import would break every student
# notebook.
#
# These previously required the retired BEGIN keyword and no @@ prefix, so they matched nothing
# and _remove_autograded_blocks() silently stripped nothing. Harmless in practice -- @@AUTOGRADED
# blocks are removed from both variants at build time, so a student notebook never contains one --
# but a strip that quietly no-ops is exactly the silent fallback this project forbids. If the
# directive syntax changes again, change it here too.
AUTOG_BEGIN_RE = re.compile(
    r'^\s*#\s*@@AUTOGRADED'
    r'(?:\s+"([^"]+)")?'
    r'(?:\s+POINTS\s+(\d+))?'
    r'(?:\s+(HINT))?\s*$'
)
AUTOG_END_RE = re.compile(r"^\s*#\s*@@AUTOGRADED\s+END(?:\s+\S+)?\s*$")

# Markdown attachment pattern: ![alt](attachment:filename)
ATTACHMENT_RE = re.compile(r"!\[([^\]]*)\]\(attachment:([^)]+)\)")

# add_image() call pattern: add_image("image_id")
ADD_IMAGE_RE = re.compile(r'add_image\s*\(\s*["\']([^"\']+)["\']\s*\)')

# add_answer() call pattern. Capture the id and deliberately stop there so optional UI-only
# keyword arguments such as allowed_formats do not hide the selector from submission generation.
ADD_ANSWER_RE = re.compile(r'add_answer\s*\(\s*["\']([^"\']+)["\']')

# add_short_answers() call pattern: add_short_answers("block_key") - one widget, many prompts.
ADD_SHORT_ANSWERS_RE = re.compile(r'add_short_answers\s*\(\s*["\']([^"\']+)["\']\s*\)')

# add_table_answer() call pattern: one schema-driven widget that saves JSON plus rendered Markdown.
ADD_TABLE_ANSWER_RE = re.compile(r'add_table_answer\s*\(\s*["\']([^"\']+)["\']\s*\)')

# Answer-selector files now live beside the notebook in Answers/. Images/ was the original name
# and remains a read-only lookup fallback so notebooks started before the rename still submit.
ANSWER_DIRECTORY_NAMES = ("Answers", "Images")
@dataclass(frozen=True)
class SubmissionQuestionHtml:
    """One logical question rendered as a standalone HTML document."""

    name: str
    points: int
    question_type: str
    html: str


def find_answer_file(notebook_directory: Path, answer_id: str, suffix: str) -> Optional[Path]:
    """Find one selector answer, preferring ``Answers`` over the legacy ``Images`` folder."""
    for directory_name in ANSWER_DIRECTORY_NAMES:
        candidate = notebook_directory / directory_name / f"{answer_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def find_answer_response(notebook_directory: Path, answer_id: str) -> AnswerResponse:
    """Find a saved answer with typed and image pages in their chosen order."""
    return read_answer_response(notebook_directory, answer_id)


def find_answer_files(notebook_directory: Path, answer_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Backward-compatible single-image/text view of an answer response."""
    response = find_answer_response(notebook_directory, answer_id)
    first_image = response.image_paths[0] if response.image_paths else None
    return first_image, response.text_path


# -----------------------------
# Notebook parsing
# -----------------------------

def load_notebook(path: Path) -> NotebookNode:
    """Load a Jupyter notebook."""
    with open(path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)


def find_cells_by_metadata_tag(nb: NotebookNode, tag: str) -> List[int]:
    """
    Find all cell indices that have a specific tag in their metadata tags list.
    
    The build tool adds these tags:
    - 'manual-answer': Cells with student answer content (from @@ANSWER blocks)
    - 'export-to-pdf': Cells whose @@APPEARS_IN names a pdf destination
    - 'export-code': Cells that should have code exported to PDF
    - 'export-output': Cells that should have output exported to PDF
    """
    result = []
    for idx, cell in enumerate(nb.cells):
        metadata = cell.get('metadata', {})
        tags = metadata.get('tags', [])
        if tag in tags:
            result.append(idx)
    return result


def find_question_cells(nb: NotebookNode) -> List[Tuple[int, str, int, str]]:
    """
    Find all cells that start questions/subparts.
    
    Returns list of (cell_index, question_name, points, question_type)
    
    Matches both the Markdown headings emitted by the current build tool and styled HTML headers
    from older builds.

    Note: A single cell may contain multiple subparts (e.g., 5.1 and 5.2 in the same cell).
    """
    questions = []
    for idx, cell in enumerate(nb.cells):
        source = cell.get('source', '')
        
        # Check markdown cells
        if cell.cell_type == 'markdown':
            # Use finditer to find ALL subparts in the cell (there may be multiple)
            found_any = False
            for match in _subpart_header_matches(source):
                subpart_num = match.group(1)  # e.g., "1.1"
                # title = match.group(2)  # e.g., "Statistical Distributions"
                points = int(match.group(3))
                ptype = match.group(4)  # "manual" or "autograded"
                # Normalize "autograded" to "auto"
                ptype = "auto" if ptype == "autograded" else ptype
                questions.append((idx, subpart_num, points, ptype))
                found_any = True
            if found_any:
                continue

            # Standalone questions (whole question graded as one item) -> "Problem N. Title (pts)".
            for match in _standalone_header_matches(source):
                number = match.group(1)          # e.g. "1"
                points = int(match.group(3))
                ptype = match.group(4)
                ptype = "auto" if ptype == "autograded" else ptype
                questions.append((idx, number, points, ptype))
                found_any = True
            if found_any:
                continue

    return questions


def get_cells_for_question(nb: NotebookNode, questions: List[Tuple[int, str, int, str]], 
                           q_idx: int) -> List[int]:
    """
    Get the cell indices that belong to a question.
    
    A question's cells extend from its start cell until:
    - The next question/subpart (exclusive)
    - A Problem header (e.g., "Problem 2. Title")
    - End of notebook
    
    Note: When multiple subparts share a cell, this returns at least that cell.
    """
    start_idx = questions[q_idx][0]
    if q_idx + 1 < len(questions):
        end_idx = questions[q_idx + 1][0]
    else:
        end_idx = len(nb.cells)
    
    # Also check for Problem headers between this question and the next
    # These act as section boundaries even though they aren't subparts
    for cell_idx in range(start_idx + 1, end_idx):
        cell = nb.cells[cell_idx]
        if cell.cell_type == 'markdown':
            source = cell.get('source', '')
            if _problem_header_match(source):
                # Found a Problem header - stop here
                end_idx = cell_idx
                break
    
    # Ensure we include at least the starting cell (for multi-subpart cells)
    if end_idx <= start_idx:
        end_idx = start_idx + 1
    
    return list(range(start_idx, end_idx))


# -----------------------------
# Code extraction for autograding
# -----------------------------

def extract_code_for_autograding(nb: NotebookNode, questions: List[Tuple[int, str, int, str]]) -> str:
    """
    Extract all code from auto-graded questions.
    
    Returns a Python string with all student code, organized by question.
    Each question is wrapped in try/except so one failing question doesn't
    crash the entire submission.
    """
    lines = [
        '"""',
        'Auto-generated code submission for Gradescope autograding.',
        'Do not edit this file directly.',
        '"""',
        '',
    ]
    
    # First, add any setup/import cells that appear before the first question
    first_q_idx = questions[0][0] if questions else len(nb.cells)
    for idx in range(first_q_idx):
        cell = nb.cells[idx]
        if cell.cell_type == 'code':
            source = cell.get('source', '').strip()
            if source:
                lines.append(f'# --- Setup Cell {idx + 1} ---')
                lines.append(source)
                lines.append('')
    
    # Now extract code from auto-graded questions
    for q_idx, (cell_idx, q_name, points, q_type) in enumerate(questions):
        if q_type != 'auto':
            continue
        
        cell_indices = get_cells_for_question(nb, questions, q_idx)
        
        # Collect all code for this question
        question_code_lines = []
        
        for cidx in cell_indices:
            cell = nb.cells[cidx]
            if cell.cell_type != 'code':
                continue
            
            source = cell.get('source', '').strip()
            if not source:
                continue
            
            # Skip autograder test blocks (they're for instructor use)
            source = _remove_autograded_blocks(source)

            # Skip cells that only contain test assertions
            if _is_only_test_cell(source):
                continue
            
            if source.strip():
                question_code_lines.append(source)
        
        if not question_code_lines:
            continue
        
        # Create a sanitized variable name from the question name for the exception
        q_var_name = q_name.replace('.', '_').replace(' ', '_').replace('-', '_')
        
        lines.append(f'# {"=" * 60}')
        lines.append(f'# Question: {q_name} ({points} points)')
        lines.append(f'# {"=" * 60}')
        lines.append('try:')
        
        # Add all question code with indentation
        for code_block in question_code_lines:
            for code_line in code_block.splitlines():
                lines.append(f'    {code_line}')
            lines.append('')
        
        # Add except block
        lines.append(f'except Exception as _err_{q_var_name}:')
        lines.append(f'    print(f"Question {q_name} code error: {{_err_{q_var_name}}}")')
        lines.append('')
    
    return '\n'.join(lines)


def _remove_autograded_blocks(source: str) -> str:
    """Remove AUTOGRADED BEGIN/END blocks from code."""
    lines = source.splitlines()
    out = []
    in_auto = False
    
    for line in lines:
        if AUTOG_BEGIN_RE.match(line):
            in_auto = True
            continue
        if AUTOG_END_RE.match(line):
            in_auto = False
            continue
        if not in_auto:
            out.append(line)
    
    return '\n'.join(out)


def _is_only_test_cell(source: str) -> bool:
    """Check if a cell contains only test assertions and prints."""
    lines = [l.strip() for l in source.splitlines() if l.strip() and not l.strip().startswith('#')]
    
    if not lines:
        return True
    
    for line in lines:
        # Allow assert statements, print statements, and pass
        if not (line.startswith('assert ') or 
                line.startswith('print(') or 
                line == 'pass' or
                line.startswith('# ')):
            return False
    
    return True


# -----------------------------
# PDF generation for manual grading
# -----------------------------

def _extract_subpart_content(source: str, q_name: str, questions: List[Tuple[int, str, int, str]], 
                              q_idx: int) -> str:
    """
    Extract only the content for a specific subpart from a cell that may contain multiple subparts.
    
    Finds the subpart header for q_name and extracts content until the next subpart header
    or end of cell.
    
    Special case: For the first subpart of a problem (e.g., 4.1), also include any
    content between the Problem header and the first subpart. This ensures that
    GRADED content with images that appears after the problem intro but before
    the first subpart is included.
    """
    # Find all subpart positions in the source
    subpart_positions = []
    for match in _subpart_header_matches(source):
        subpart_num = match.group(1)
        subpart_positions.append((match.start(), match.end(), subpart_num))
    
    if not subpart_positions:
        return source  # No styled subparts found, return full content
    
    # Find our subpart
    our_start = None
    our_end = len(source)
    
    for i, (start, end, num) in enumerate(subpart_positions):
        if num == q_name:
            # Start from the subpart header by default
            our_start = start
            
            # End at the next subpart (if any)
            if i + 1 < len(subpart_positions):
                our_end = subpart_positions[i + 1][0]
            
            # Special case: if this is the FIRST subpart in the cell (i == 0),
            # include any content between the Problem header and this subpart
            # (this includes intro text and images before the first subpart)
            if i == 0:
                # Look for the Problem header
                problem_match = _problem_header_match(source)
                if problem_match:
                    # Include content from after the Problem header (but still include the subpart header)
                    our_start = problem_match.end()
                else:
                    # No Problem header - include from the beginning
                    our_start = 0
            break
    
    if our_start is None:
        return source  # Our subpart not found, return full content
    
    return source[our_start:our_end].strip()


def extract_manual_content_html(
    nb: NotebookNode,
    questions: List[Tuple[int, str, int, str]],
    notebook_path: Path,
    *,
    hide_answers: bool = False,
    document_title: Optional[str] = None,
    document_subtitle: str = "Student submission for manual grading",
    split_questions: bool = False,
) -> str | List[SubmissionQuestionHtml]:
    """
    Generate submission HTML for manual answers and autograded audit records.

    Every declared question is represented, even when its response is missing. Auto-graded code
    and output appear when their cells carry the build's ``export-to-pdf`` tags.

    When ``hide_answers`` is true, answer material is made invisible with CSS ``visibility``.
    Invisible content still participates in layout, so the resulting blank Gradescope template
    has the same pagination as a PDF generated from the populated solution notebook. Question
    headers and individual @@SHORT_ANSWER prompts remain visible. With ``split_questions=True``,
    return one standalone HTML document per logical question so the PDF writer can give each a
    single variable-height physical page.
    """
    # Filter to only manually-graded questions
    manual_questions = [(q_idx, q) for q_idx, q in enumerate(questions) if q[3] == 'manual']
    
    html_parts = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '<meta charset="utf-8">',
        '<style>',
        '@page { size: A4; margin: 12mm; } /* PSET_PAGE_SIZE */',
        'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; ',
        '       font-size: 11pt; line-height: 1.45; ',
        '       max-width: 800px; margin: 40px auto; padding: 20px; }',
        'h1, h2, h3 { color: #333; }',
        '.question-heading { color: #7030A0; font-weight: bold; margin: 0; ',
        '                    page-break-after: avoid; break-after: avoid; }',
        'h2.question-heading { font-size: 1.5em; text-align: center; }',
        'h3.question-heading { font-size: 1.1em; }',
        '.question-heading .question-heading-points { font-weight: normal; }',
        '.question { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; ',
        '            page-break-before: always; }',
        '.question:first-of-type { page-break-before: avoid; }',
        '.question-header { background: #f5f5f5; margin: -20px -20px 20px -20px; ',
        '                   padding: 15px 20px; border-radius: 8px 8px 0 0; }',
        '.code { background: #f8f8f8; padding: 10px; border-radius: 4px; ',
        '        font-family: monospace; white-space: pre-wrap; overflow-x: auto; }',
        '.markdown { margin: 10px 0; }',
        'img { max-width: 100%; height: auto; margin: 10px 0; }',
        '.cell-output { background: #fafafa; border-left: 3px solid #4CAF50; padding: 10px; margin: 10px 0; }',
        '.question-kind { color: #666; font-size: 0.85em; margin-left: 10px; }',
        '.no-response { border: 2px dashed #cc0000; padding: 20px; color: #cc0000; ',
        '               background: #fff0f0; text-align: center; }',
        '.answer-image-page { margin: 14px 0 24px; break-inside: avoid; }',
        '.answer-image-page img { display: block; max-width: 100%; max-height: 235mm; ',
        '                         width: auto; height: auto; object-fit: contain; }',
        '.answer-image-page-label { color: #555; font-size: 0.85em; font-weight: bold; ',
        '                           margin-bottom: 4px; }',
        '.answer-text-page { margin: 14px 0 24px; }',
        'hr { border: none; border-top: 2px solid #eee; margin: 30px 0; }',
        '.math-inline { font-style: italic; }',
        '.math-block { display: block; text-align: center; margin: 1em 0; }',
        # Math SVGs must escape the global img rule above: it forces height:auto (discarding the
        # intrinsic point height that makes math match the body text) and adds a 10px block margin.
        '.math-svg { margin: 0; max-width: none; height: auto; }',
        # Display math keeps its intrinsic size until it would leave the page, then
        # scales down to the column. Inline math stays unbounded so it matches the prose.
        '.math-block .math-svg { max-width: 100%; }',
        '.math-svg-inline { display: inline; }',
        '.math-error { color: #cc0000; background: #fff0f0; border: 1px solid #cc0000; ',
        '              padding: 1px 4px; border-radius: 3px; }',
        # A blank Gradescope template is rendered from the populated solution so its pagination
        # exactly matches the solution's answers_<email-id>.pdf. visibility:hidden suppresses painting
        # while preserving every answer's dimensions; display:none would collapse the layout.
        '.manual-grading-template .question > :not(.question-header) { visibility: hidden; }',
        '.manual-grading-template .question > .manual-prompt, ',
        '.manual-grading-template .question > .manual-prompt * { visibility: visible; }',
        '</style>',
        '</head>',
        '<body class="manual-grading-template">' if hide_answers else '<body>',
    ]
    html_shell_prefix = list(html_parts)
    
    # Get notebook title from first markdown cell or filename
    title = document_title or notebook_path.stem.replace('_', ' ')
    html_parts.append(f'<h1>{_escape_html(title)}</h1>')
    html_parts.append(f'<p><em>{_escape_html(document_subtitle)}</em></p>')
    html_parts.append('<hr>')
    
    # Get attachments from the notebook
    attachments = _collect_all_attachments(nb)
    
    # Collect all cells to render with their metadata, then process in document order
    # Each entry: (cell_idx, cell_type, q_name, q_points)
    # cell_type is 'manual-answer' or 'export-to-pdf'
    cells_to_render = []
    
    # Find all cells with manual-answer tag
    answer_cell_indices = set(find_cells_by_metadata_tag(nb, 'manual-answer'))

    def points_for_question(q_name: str) -> int:
        """Points of the subpart a widget cell belongs to, 0 if the name is not a question.

        A widget cell (add_image / add_answer / add_short_answers) carries no points of its
        own -- its only link back to its subpart is the question-X.Y tag -- so the value has
        to be looked up by name. Without this the question header in the PDF loses its
        "(N points)" annotation for any subpart whose only content is a widget cell.
        """
        for _cell_idx, name, points, _q_type in questions:
            if name == q_name:
                return points
        return 0

    # Add manual-answer cells with their question info
    for q_idx, (cell_idx, q_name, points, q_type) in manual_questions:
        cell_indices = get_cells_for_question(nb, questions, q_idx)
        for cidx in cell_indices:
            if cidx in answer_cell_indices:
                cells_to_render.append((cidx, 'manual-answer', q_name, points, None))
    
    # Add export-to-pdf cells (excluding those already in answer cells)
    export_cells = find_cells_by_metadata_tag(nb, 'export-to-pdf')
    for cidx in export_cells:
        if cidx not in answer_cell_indices:
            cell = nb.cells[cidx]
            metadata = cell.get('metadata', {})
            tags = metadata.get('tags', [])
            
            # Find question name from question-X.Y tag
            q_name = "Code Output"
            q_points = 0
            for tag in tags:
                if tag.startswith('question-'):
                    q_name = tag[9:]  # strip "question-" prefix
                    # Look up points from questions list
                    for q_cell_idx, name, points, q_type in questions:
                        if name == q_name:
                            q_points = points
                            break
                    break
            cells_to_render.append((cidx, 'export-to-pdf', q_name, q_points, None))
    
    # Add image, free-form answer, short-answer, and table-answer widget cells.
    missing_answers = []
    # @@SHORT_ANSWER prompt data (questions + ids), written by the build into Files/.
    short_answer_prompts_data = {}
    sa_prompts_path = notebook_path.parent / 'Files' / 'short_answer_prompts.json'
    if sa_prompts_path.exists():
        import json as _json
        try:
            short_answer_prompts_data = _json.loads(sa_prompts_path.read_text(encoding='utf-8'))
        except Exception:
            short_answer_prompts_data = {}

    table_answer_schemas = {}
    table_schemas_path = notebook_path.parent / 'Files' / 'table_answer_schemas.json'
    if table_schemas_path.exists():
        import json as _json
        try:
            table_answer_schemas = _json.loads(
                table_schemas_path.read_text(encoding='utf-8'))
        except Exception:
            table_answer_schemas = {}

    for cidx, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            source = cell.get('source', '')

            # Check for add_table_answer("id") before the generic answer selector. Its Markdown
            # rendering enters the PDF like a typed answer; JSON is additionally checked so a
            # partially completed table is reported rather than silently appearing complete.
            match = ADD_TABLE_ANSWER_RE.search(source)
            if match:
                answer_id = match.group(1)
                text_path = find_answer_file(notebook_path.parent, answer_id, '.md')
                json_path = find_answer_file(notebook_path.parent, answer_id, '.json')
                schema = table_answer_schemas.get(answer_id, {})
                missing_cells = None
                if json_path is not None and schema:
                    try:
                        import json as _json
                        payload = _json.loads(json_path.read_text(encoding='utf-8'))
                        responses = payload.get('responses', {})
                        missing_cells = sum(
                            not str(responses.get(row.get('id', ''), {}).get(
                                column.get('id', ''), '')).strip()
                            for row in schema.get('rows', [])
                            for column in schema.get('columns', [])
                        )
                    except Exception:
                        missing_cells = None
                if text_path is None or json_path is None or missing_cells is None or missing_cells:
                    missing_answers.append((answer_id, 'table'))
                tags = cell.get('metadata', {}).get('tags', [])
                q_name = answer_id
                for tag in tags:
                    if tag.startswith('question-'):
                        q_name = tag[len('question-'):]
                        break
                q_points = points_for_question(q_name)
                cells_to_render.append((
                    cidx, 'table-answer', q_name, q_points, answer_id,
                    text_path is not None, missing_cells,
                ))
                continue

            # Check for add_short_answers("key") - one widget cell holding many prompts.
            match = ADD_SHORT_ANSWERS_RE.search(source)
            if match:
                block_key = match.group(1)
                prompts = short_answer_prompts_data.get(block_key, [])
                tags = cell.get('metadata', {}).get('tags', [])
                q_name = block_key
                for tag in tags:
                    if tag.startswith('question-'):
                        q_name = tag[len('question-'):]
                        break
                q_points = points_for_question(q_name)
                for prompt in prompts:
                    pid = prompt.get('id', '')
                    image_path, text_path = find_answer_files(notebook_path.parent, pid)
                    if image_path is None and text_path is None:
                        missing_answers.append((pid, 'answer'))
                cells_to_render.append((cidx, 'short-answers', q_name, q_points, prompts))
                continue

            # Check for add_image() call
            match = ADD_IMAGE_RE.search(source)
            if match:
                image_id = match.group(1)
                # Check if image file exists
                image_path = find_answer_file(notebook_path.parent, image_id, '.png')
                image_exists = image_path is not None
                if not image_exists:
                    missing_answers.append((image_id, 'image'))
                # Find question context from question-X.Y tag
                q_name = image_id  # default to image_id
                metadata = cell.get('metadata', {})
                tags = metadata.get('tags', [])
                for tag in tags:
                    if tag.startswith('question-'):
                        q_name = tag[len('question-'):]  # Extract "1.1" from "question-1.1"
                        break
                q_points = points_for_question(q_name)
                cells_to_render.append((cidx, 'add-image', q_name, q_points, image_id, image_exists, None))
                continue  # Don't also check for add_answer in the same cell
            
            # Check for add_answer() call (supports both image and text)
            match = ADD_ANSWER_RE.search(source)
            if match:
                answer_id = match.group(1)
                response = find_answer_response(notebook_path.parent, answer_id)
                if not response.pages:
                    missing_answers.append((answer_id, 'answer'))
                # Find question context from question-X.Y tag
                q_name = answer_id  # default to answer_id
                metadata = cell.get('metadata', {})
                tags = metadata.get('tags', [])
                for tag in tags:
                    if tag.startswith('question-'):
                        q_name = tag[len('question-'):]  # Extract "1.1" from "question-1.1"
                        break
                q_points = points_for_question(q_name)
                cells_to_render.append((
                    cidx, 'add-answer', q_name, q_points, answer_id,
                    response,
                ))
    
    # Warn about missing answers
    if missing_answers:
        print(f"\n⚠️  WARNING: {len(missing_answers)} answer(s) not provided:")
        for answer_id, answer_type in missing_answers:
            if answer_type == 'image':
                print(f"   - {answer_id} (run add_image('{answer_id}') to add)")
            elif answer_type == 'table':
                print(f"   - {answer_id} (run add_table_answer('{answer_id}') and complete every cell)")
            else:
                print(f"   - {answer_id} (run add_answer('{answer_id}') to add)")
    
    # Sort by cell index to maintain document order
    cells_to_render.sort(key=lambda x: x[0])

    def render_entry_inner(entry) -> str:
        """Render one entry's content WITHOUT the outer .question div / header, so several
        entries that belong to the same question (e.g. the prompts + text boxes of a
        @@SHORT_ANSWER block) can share a single question block / PDF page."""
        cidx, cell_type = entry[0], entry[1]
        cell = nb.cells[cidx]
        source = cell.get('source', '')
        tags = cell.get('metadata', {}).get('tags', [])
        parts: List[str] = []

        if cell_type == 'add-image':
            image_id, image_exists = entry[4], entry[5]
            image_path = find_answer_file(notebook_path.parent, image_id, '.png')
            if image_exists:
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                parts.append(f'<img src="data:image/png;base64,{image_data}" alt="{image_id}">')
            else:
                parts.append('<div style="border: 2px dashed #cc0000; padding: 20px; margin: 10px 0; '
                             'background-color: #fff0f0; color: #cc0000; text-align: center;">')
                parts.append('<strong>⚠️ Image not provided</strong><br>')
                parts.append(f'<span style="font-size: 0.9em;">Run add_image("{image_id}") to add your image</span>')
                parts.append('</div>')

        elif cell_type == 'add-answer':
            answer_id, response = entry[4], entry[5]
            for page_number, page in enumerate(response.pages, start=1):
                if page.kind == "text":
                    content = page.path.read_text(encoding='utf-8')
                    if content.startswith('---'):
                        split = content.split('---', 2)
                        if len(split) >= 3:
                            content = split[2].strip()
                    parts.append(
                        '<div class="markdown text-answer answer-text-page">'
                        f'<div class="answer-image-page-label">Page {page_number}</div>'
                        f'{_markdown_to_html(content, {}, cell)}</div>')
                else:
                    with open(page.path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                    parts.append(
                        '<div class="answer-image-page">'
                        f'<div class="answer-image-page-label">Page {page_number}</div>'
                        f'<img src="data:image/png;base64,{image_data}" '
                        f'alt="{answer_id}, page {page_number}">'
                        '</div>'
                    )
            if not response.pages:
                parts.append('<div style="border: 2px dashed #cc0000; padding: 20px; margin: 10px 0; '
                             'background-color: #fff0f0; color: #cc0000; text-align: center;">')
                parts.append('<strong>⚠️ Answer not provided</strong><br>')
                parts.append(f'<span style="font-size: 0.9em;">Run add_answer("{answer_id}") to add your answer</span>')
                parts.append('</div>')

        elif cell_type == 'table-answer':
            answer_id, text_exists, missing_cells = entry[4], entry[5], entry[6]
            text_path = find_answer_file(notebook_path.parent, answer_id, '.md')
            if text_exists and text_path is not None:
                content = text_path.read_text(encoding='utf-8')
                if content.startswith('---'):
                    split = content.split('---', 2)
                    if len(split) >= 3:
                        content = split[2].strip()
                parts.append(
                    f'<div class="markdown text-answer">'
                    f'{_markdown_to_html(content, {}, cell)}</div>')
                if missing_cells:
                    parts.append(
                        '<div style="color:#cc0000; font-size:0.9em; margin-top:4px;">'
                        f'&#9888; {missing_cells} table cell(s) were left blank.</div>')
            else:
                parts.append('<div style="border: 2px dashed #cc0000; padding: 20px; margin: 10px 0; '
                             'background-color: #fff0f0; color: #cc0000; text-align: center;">')
                parts.append('<strong>⚠️ Table answer not provided</strong><br>')
                parts.append(
                    f'<span style="font-size: 0.9em;">Run add_table_answer("{answer_id}") '
                    'and complete every cell</span>')
                parts.append('</div>')

        elif cell_type == 'manual-answer':
            if cell.cell_type == 'markdown':
                # Released notebooks use colored bars to distinguish answer regions while a
                # student is working. The submission PDF already encloses every response in a
                # bordered question box, so those generated bars are redundant and can force an
                # otherwise complete answer onto an extra page.
                cleaned = _strip_generated_manual_answer_bars(
                    _normalize_cell_source(source)
                )
                parts.append(f'<div class="markdown">{_markdown_to_html(cleaned, attachments, cell)}</div>')
            elif cell.cell_type == 'code':
                cleaned = _normalize_cell_source(source)
                if cleaned.strip():
                    parts.append(f'<div class="code">{_escape_html(cleaned)}</div>')
                outputs_html = _render_cell_outputs(cell)
                if outputs_html:
                    parts.append(outputs_html)

        elif cell_type == 'export-to-pdf' and cell.cell_type == 'code':
            if 'export-code' in tags:
                cleaned = _normalize_cell_source(source)
                if cleaned.strip():
                    parts.append(f'<div class="code">{_escape_html(cleaned)}</div>')
            if 'export-output' in tags:
                outputs_html = _render_cell_outputs(cell)
                if outputs_html:
                    parts.append(outputs_html)

        elif cell_type == 'short-answers':
            # One @@SHORT_ANSWER widget cell: render each prompt (question label) + its answer image.
            for prompt in entry[4]:
                pid = prompt.get('id', '')
                points = prompt.get('points')
                points_note = ''
                if points is not None:
                    points_note = f" ({points} {'pt' if points == 1 else 'pts'})"
                # Emphasise only the (a)/(b) label. Wrapping the question in ** too would invert
                # the emphasis of any question containing its own **bold**, since markdown closes
                # the outer span at the inner marker.
                heading = f"**{prompt.get('label', '')}** {prompt.get('question', '')}{points_note}"
                parts.append(
                    f'<div class="markdown manual-prompt">'
                    f'{_markdown_to_html(heading, {}, cell)}</div>'
                )

                image_path, text_path = find_answer_files(notebook_path.parent, pid)
                # Text first -- see the matching note in the add-answer branch above.
                if text_path is not None:
                    content = text_path.read_text(encoding='utf-8')
                    if content.startswith('---'):
                        split = content.split('---', 2)
                        if len(split) >= 3:
                            content = split[2].strip()
                    parts.append(f'<div class="markdown text-answer">{_markdown_to_html(content, {}, cell)}</div>')
                elif image_path is not None:
                    with open(image_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                    parts.append(f'<img src="data:image/png;base64,{image_data}" alt="{pid}">')
                else:
                    parts.append('<div style="border: 2px dashed #cc0000; padding: 12px; margin: 6px 0 16px 0; '
                                 'background-color: #fff0f0; color: #cc0000;">⚠️ Answer not provided</div>')

        return '\n'.join(parts)

    entries_by_question: Dict[str, List[tuple]] = {name: [] for _, name, _, _ in questions}
    for entry in cells_to_render:
        q_name = entry[2]
        if q_name not in entries_by_question:
            raise ValueError(
                f"PDF content at notebook cell {entry[0] + 1} is tagged for unknown question "
                f"{q_name!r}. Rebuild the assignment so every exported cell has a valid "
                "question-X.Y tag."
            )
        entries_by_question[q_name].append(entry)

    question_blocks: List[SubmissionQuestionHtml] = []
    for _cell_idx, q_name, q_points, q_type in questions:
        group = entries_by_question[q_name]
        inner = [html for html in (render_entry_inner(entry) for entry in group) if html.strip()]
        if not inner:
            inner = ['<div class="no-response">No response was provided for this question.</div>']
        type_label = "Autograded — audit record" if q_type == "auto" else "Manual"
        block = [
            '<div class="question">',
            '<div class="question-header">',
            f'<strong>Question {q_name}</strong> ({q_points} points)',
            f'<span class="question-kind">{type_label}</span>',
            '</div>',
            *inner,
            '</div>',
        ]
        block_html = '\n'.join(block)
        question_blocks.append(SubmissionQuestionHtml(
            name=q_name,
            points=q_points,
            question_type=q_type,
            html=block_html,
        ))
        html_parts.append(block_html)

    if split_questions:
        documents = []
        for index, question in enumerate(question_blocks):
            prefix = list(html_shell_prefix)
            if index == 0:
                prefix.extend([
                    f'<h1>{_escape_html(title)}</h1>',
                    f'<p><em>{_escape_html(document_subtitle)}</em></p>',
                    '<hr>',
                ])
            documents.append(SubmissionQuestionHtml(
                name=question.name,
                points=question.points,
                question_type=question.question_type,
                html='\n'.join([*prefix, question.html, '</body>', '</html>']),
            ))
        return documents

    html_parts.extend(['</body>', '</html>'])
    return '\n'.join(html_parts)


def _collect_all_attachments(nb: NotebookNode) -> Dict[str, str]:
    """Collect all cell attachments as base64 data URIs."""
    attachments = {}
    for cell in nb.cells:
        cell_attachments = cell.get('attachments', {})
        for name, data in cell_attachments.items():
            # data is a dict like {'image/png': 'base64data...'}
            for mime_type, b64data in data.items():
                attachments[name] = f'data:{mime_type};base64,{b64data}'
    return attachments


def _normalize_cell_source(source: str) -> str:
    """Strip a trailing newline and normalize line endings on a cell's source.

    The PDF renders code in a `white-space: pre-wrap` block, so a trailing newline shows up as a
    blank line at the bottom of the block. This normalization used to happen as a side effect of
    the INSTRUCTOR/STUDENT block strippers (they rebuilt the source with '\\n'.join(splitlines()));
    those matched retired directives and were deleted, so the normalization is now explicit.
    """
    return '\n'.join(source.splitlines())


# Exact wrappers inserted by release_assignment.manual_answer_bar(). Matching complete lines keeps
# an instructor-authored horizontal rule in an answer intact.
_GENERATED_MANUAL_ANSWER_BARS = {
    '<hr style="border: none; height: 2px; background: linear-gradient(to right, '
    '#7030A0, #f3e8ff);"/>',
    '<hr style="border: none; height: 2px; background: linear-gradient(to right, '
    '#2e7d32, #e8f5e9);"/>',
}


def _strip_generated_manual_answer_bars(source: str) -> str:
    """Remove release-generated answer delimiters from submission-PDF markdown."""
    return '\n'.join(
        line for line in source.splitlines()
        if line.strip() not in _GENERATED_MANUAL_ANSWER_BARS
    ).strip()


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;'))


def _escape_math_alt(latex: str) -> str:
    """Escape LaTeX for an ``alt`` attribute that markdown will run over afterwards.

    _markdown_to_html converts math BEFORE calling the markdown library, so the alt text of every
    math <img> is still parsed as markdown. Square brackets are the problem: ``[C](t)`` is link
    syntax, and the ``<a href="t">`` markdown substitutes closes the alt attribute early, spilling
    the rest of the expression onto the page as visible text. Encoding the brackets leaves nothing
    for the link parser to match; the HTML parser decodes them again afterwards.
    """
    return (_escape_html(latex)
            .replace('[', '&#91;')
            .replace(']', '&#93;'))


# Math rendering settings. The SVG that matplotlib emits carries its own intrinsic width/height
# in points at the requested font size, so it drops into the page at the correct optical size with
# no scaling. MATH_RENDER_DPI only sets the rendering resolution; math_to_image always calculates
# and returns the baseline depth at 72 dpi, so that value is already in points.
# These track the body font-size set in the stylesheet, so inline math sits at the same optical
# size as the prose around it. Change them together.
MATH_BODY_FONTSIZE_POINTS = 11
MATH_DISPLAY_FONTSIZE_POINTS = 14
MATH_RENDER_DPI = 200
MATH_TEXT_COLOR = '#111111'

# LaTeX fragments that failed to render, collected during a run so the failures can be reported
# together at the end instead of one warning per occurrence.
_math_render_failures: List[str] = []


# matplotlib's mathtext understands a SUBSET of LaTeX, so several commands an instructor would
# reasonably type are rejected outright. Each entry below was confirmed by testing: the key is
# rejected by mathtext, the replacement renders and means the same thing.
#
# Deliberately NOT aliased: \begin{cases}...\end{cases} has no mathtext equivalent, so it still
# fails loudly rather than being silently mangled into something that means something else.
_MATH_COMMAND_ALIASES = {
    # Comparison shorthands. Standard LaTeX accepts both spellings; mathtext only the long one.
    r'\le':        r'\leq',
    r'\ge':        r'\geq',
    # Arrows.
    r'\gets':      r'\leftarrow',
    r'\iff':       r'\Leftrightarrow',
    r'\implies':   r'\Rightarrow',
    # Upright text. \mathrm is the closest mathtext equivalent of both.
    r'\textrm':    r'\mathrm',
    r'\mbox':      r'\mathrm',
    # Text-style fraction -> ordinary fraction (mathtext has no \tfrac).
    r'\tfrac':     r'\frac',
}

# Manual delimiter sizing has no mathtext equivalent. Dropping the prefix leaves the plain
# delimiter, which renders correctly at its natural size -- a cosmetic degradation, not a
# meaning change.
_MATH_SIZING_PREFIXES = (r'\bigg', r'\Bigg', r'\big', r'\Big')

# Negative thin space; mathtext has no equivalent, and spacing carries no meaning here.
_MATH_DROPPED_COMMANDS = (r'\!',)


def _normalize_latex_for_mathtext(latex: str) -> str:
    """Rewrite common LaTeX that mathtext rejects into the equivalent it accepts.

    Instructors write ordinary LaTeX; mathtext implements a subset of it. Without this, an
    expression as everyday as ``n \\le 5`` renders as a red error placeholder in the PDF.

    Uses a word boundary on the command name so ``\\le`` does not corrupt ``\\leq`` or
    ``\\leftarrow`` -- a naive string replace would turn ``\\leftarrow`` into ``\\leqftarrow``.
    """
    for command, replacement in _MATH_COMMAND_ALIASES.items():
        latex = re.sub(re.escape(command) + r'(?![a-zA-Z])', replacement.replace('\\', r'\\'), latex)
    for prefix in _MATH_SIZING_PREFIXES:
        latex = re.sub(re.escape(prefix) + r'(?![a-zA-Z])', '', latex)
    for command in _MATH_DROPPED_COMMANDS:
        latex = latex.replace(command, '')
    return latex


# \cancel{...} (amsmath's cancel package) strikes through a term. mathtext has no such command, so
# it is rendered here instead: the term is laid out normally and a diagonal stroke is drawn over
# the glyphs that belong to it.
_CANCEL_OPENER_RE = re.compile(r'\\cancel(?![a-zA-Z])\s*\{')

# Placeholder that stands in for a cancelled term while locating it. Chosen because mathtext lays
# it out as one glyph and it cannot appear in a real expression.
_CANCEL_SENTINEL = r'\mathrm{\#}'


def _split_cancel_groups(latex: str) -> List[Tuple[int, int, str]]:
    """Return (start, end, inner) for every \\cancel{...}, matching nested braces.

    Raises ValueError on unbalanced braces so the expression falls through to the visible
    unrenderable-math placeholder rather than rendering with a stroke in the wrong place.
    """
    groups = []
    for match in _CANCEL_OPENER_RE.finditer(latex):
        depth, index = 1, match.end()
        while depth and index < len(latex):
            if latex[index] == '\\':
                index += 2
                continue
            depth += {'{': 1, '}': -1}.get(latex[index], 0)
            index += 1
        if depth:
            raise ValueError(f"unbalanced braces after \\cancel at position {match.start()}")
        groups.append((match.start(), index, latex[match.end():index - 1]))
    if any(_CANCEL_OPENER_RE.search(inner) for _, _, inner in groups):
        raise ValueError("nested \\cancel is not supported")
    return groups


def _cancel_stroke_boxes(latex: str, font_size: float) -> Tuple[str, List[Tuple[float, float, float, float]]]:
    """Strip \\cancel from ``latex`` and return the plain expression plus one box per cancelled term.

    Each box is (left, bottom, right, top) in points, measured from the left edge of the layout
    and up from its baseline -- the frame mathtext.math_to_image draws in. A glyph belongs to a
    cancelled term when it disappears once that term is replaced by a placeholder; laying the
    expression out once per term identifies its glyphs without having to parse LaTeX.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.ft2font import LoadFlags
    from matplotlib.mathtext import MathTextParser

    groups = _split_cancel_groups(latex)
    if not groups:
        return latex, []

    def plain_with(replacement_index: Optional[int]) -> str:
        expression = latex
        for group_index in reversed(range(len(groups))):
            start, end, inner = groups[group_index]
            inner = _CANCEL_SENTINEL if group_index == replacement_index else inner
            expression = expression[:start] + '{' + inner + '}' + expression[end:]
        return _normalize_latex_for_mathtext(expression)

    parser = MathTextParser('path')
    prop = FontProperties(size=font_size)

    def glyph_keys_and_boxes(expression: str):
        layout = parser.parse(f'${expression}$', prop=prop)
        entries = []
        for font, size, code, _, x, y in layout.glyphs:
            font.set_size(size, 72)
            left, bottom, right, top = (value / 64.0 for value in
                                        font.load_char(code, flags=LoadFlags.NO_HINTING).bbox)
            entries.append(((code, round(float(size), 3)), (x + left, y + bottom, x + right, y + top)))
        return entries

    from difflib import SequenceMatcher

    sentinel_code = ord('#')
    plain = plain_with(None)
    full = glyph_keys_and_boxes(plain)
    full_keys = [key for key, _ in full]
    boxes = []
    for group_index in range(len(groups)):
        reduced = [key for key, _ in glyph_keys_and_boxes(plain_with(group_index))
                   if key[0] != sentinel_code]
        # The glyphs of the full layout with no counterpart in the reduced one are exactly the
        # ones the cancelled term contributed. A diff, not a positional walk, because removing
        # the term shifts everything after it.
        matched = set()
        for block in SequenceMatcher(None, full_keys, reduced, autojunk=False).get_matching_blocks():
            matched.update(range(block.a, block.a + block.size))
        cancelled = [box for index, (_, box) in enumerate(full) if index not in matched]
        if not cancelled:
            raise ValueError(f"\\cancel{{{groups[group_index][2]}}} has no visible glyphs to strike")
        boxes.append((min(b[0] for b in cancelled), min(b[1] for b in cancelled),
                      max(b[2] for b in cancelled), max(b[3] for b in cancelled)))
    return plain, boxes


def _add_cancel_strokes(svg: str, boxes: List[Tuple[float, float, float, float]],
                        baseline_points: float) -> str:
    """Draw one bottom-left to top-right stroke per box into a math_to_image SVG."""
    padding = 1.0
    strokes = ''.join(
        f'<line x1="{left - padding:.2f}" y1="{baseline_points - bottom + padding:.2f}" '
        f'x2="{right + padding:.2f}" y2="{baseline_points - top - padding:.2f}" '
        f'stroke="{MATH_TEXT_COLOR}" stroke-width="0.8" stroke-linecap="round"/>'
        for left, bottom, right, top in boxes)
    closing = svg.rindex('</svg>')
    return svg[:closing] + strokes + svg[closing:]


def _render_latex_to_svg_image(latex: str, *, display: bool) -> str:
    """Render one LaTeX fragment to an inline <img> holding a data-URI SVG.

    matplotlib's mathtext lays the expression out as vector paths, which WeasyPrint draws
    faithfully. This replaces a MathML backend that WeasyPrint silently flattened -- it has no
    MathML support, so superscripts and fraction bars collapsed onto a single baseline.

    ``math_to_image`` calculates layout at 72 dpi and returns the depth of the expression below
    its baseline in points, regardless of the requested output dpi. Lowering the inline image by
    that depth aligns the expression's internal baseline with the surrounding text's baseline.
    """
    from io import BytesIO

    from matplotlib import mathtext, rc_context
    from matplotlib.font_manager import FontProperties

    font_size = MATH_DISPLAY_FONTSIZE_POINTS if display else MATH_BODY_FONTSIZE_POINTS
    plain_latex, cancel_boxes = _cancel_stroke_boxes(latex, font_size)
    buffer = BytesIO()
    # math_to_image creates a Figure internally. Its default opaque white figure patch becomes
    # a white rectangle in the SVG, which is conspicuous on colored answer cards. These rcParams
    # make that patch transparent while leaving the math glyph color unchanged.
    with rc_context({
        'figure.facecolor': 'none',
        'savefig.facecolor': 'none',
        'savefig.edgecolor': 'none',
        'savefig.transparent': True,
    }):
        depth_points = mathtext.math_to_image(
            f"${_normalize_latex_for_mathtext(plain_latex)}$",
            buffer,
            prop=FontProperties(size=font_size),
            dpi=MATH_RENDER_DPI,
            format='svg',
            color=MATH_TEXT_COLOR,
        )
    svg = buffer.getvalue().decode('utf-8')
    if cancel_boxes:
        # math_to_image draws in points with the baseline at (height - depth) from the top.
        height_points = float(re.search(r'viewBox="0 0 [\d.]+ ([\d.]+)"', svg).group(1))
        svg = _add_cancel_strokes(svg, cancel_boxes, height_points - (depth_points or 0.0))
    encoded_svg = base64.b64encode(svg.encode('utf-8')).decode('ascii')
    source = f'data:image/svg+xml;base64,{encoded_svg}'

    if display:
        return f'<img class="math-svg" src="{source}" alt="{_escape_math_alt(latex)}">'

    depth_points = depth_points or 0.0
    return (f'<img class="math-svg math-svg-inline" src="{source}" '
            f'style="vertical-align: -{depth_points:.2f}pt;" alt="{_escape_math_alt(latex)}">')


def _render_failed_math(latex: str, *, display: bool, error: Exception) -> str:
    """Render a visible, impossible-to-miss placeholder for LaTeX that would not compile.

    Deliberately non-fatal: a single malformed expression must not abort a student's whole
    submission. The failure is loud in two places instead -- a red box in the PDF itself, and a
    summary printed by generate_pset_submission() listing every fragment that failed.
    """
    _math_render_failures.append(f"{latex}  ({type(error).__name__}: {error})")
    tag = 'div' if display else 'span'
    return (f'<{tag} class="math-error" title="{_escape_html(str(error))}">'
            f'⚠️ unrenderable math: <code>{_escape_html(latex)}</code></{tag}>')


def _convert_latex_math(text: str) -> str:
    """
    Convert LaTeX math expressions to inline SVG images for PDF rendering.

    Handles:
    - Display math: $$ ... $$ or \\[ ... \\]
    - Inline math: $ ... $ or \\( ... \\)

    Applied to every markdown cell in the submission, so instructor question text, model answers
    and student-typed answers all render math through the same path.

    Fenced blocks and inline code spans are masked out first. This pass runs BEFORE the markdown
    converter, so without masking, a `$...$` written inside backticks -- exactly how you show a
    student what LaTeX to type -- was replaced by an <img> tag, and the code span then rendered
    that tag as literal text. A page of base64 in the middle of the instructions is the result.
    """
    protected: List[str] = []

    def protect(match):
        protected.append(match.group(0))
        return f"\x00CODE{len(protected) - 1}\x00"

    # Fenced blocks first: a ``` block may itself contain single-backtick spans.
    text = re.sub(r'```.*?```', protect, text, flags=re.DOTALL)
    text = re.sub(r'``[^`]+``|`[^`\n]+`', protect, text)

    def replace_display(match):
        latex = match.group(1).strip()
        try:
            image = _render_latex_to_svg_image(latex, display=True)
        except Exception as error:
            return _render_failed_math(latex, display=True, error=error)
        return f'<div class="math-block">{image}</div>'

    def replace_inline(match):
        latex = match.group(1).strip()
        try:
            return _render_latex_to_svg_image(latex, display=False)
        except Exception as error:
            return _render_failed_math(latex, display=False, error=error)

    # Display math: $$...$$ (must come first to avoid matching as two inline $)
    text = re.sub(r'\$\$(.+?)\$\$', replace_display, text, flags=re.DOTALL)

    # Display math: \[...\]
    text = re.sub(r'\\\[(.+?)\\\]', replace_display, text, flags=re.DOTALL)

    # Inline math: $...$ (but not $$)
    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', replace_inline, text)

    # Inline math: \(...\)
    text = re.sub(r'\\\((.+?)\\\)', replace_inline, text)

    # Put the code back exactly as it was written.
    def restore(match):
        return protected[int(match.group(1))]

    return re.sub(r'\x00CODE(\d+)\x00', restore, text)


def _markdown_to_html(source: str, attachments: Dict[str, str], cell: NotebookNode) -> str:
    """
    Convert markdown to HTML with embedded images.
    
    Handles:
    - Attachment images: ![alt](attachment:filename)
    - Cell-level attachments
    - Markdown formatting including tables
    - LaTeX math expressions
    """
    # First, convert LaTeX math to MathML or formatted text
    source = _convert_latex_math(source)
    
    # Try to use a proper markdown library if available
    try:
        import markdown
        from markdown.extensions.tables import TableExtension
        from markdown.extensions.fenced_code import FencedCodeExtension
        
        # Get cell-level attachments
        cell_attachments = cell.get('attachments', {})
        
        # Replace attachment references with data URIs before markdown processing
        def replace_attachment(match):
            alt = match.group(1)
            filename = match.group(2)
            
            # Check cell attachments first
            if filename in cell_attachments:
                data = cell_attachments[filename]
                for mime_type, b64data in data.items():
                    return f'<img src="data:{mime_type};base64,{b64data}" alt="{alt}">'
            
            # Fall back to global attachments
            if filename in attachments:
                return f'<img src="{attachments[filename]}" alt="{alt}">'
            
            return f'<em>[Image not found: {filename}]</em>'
        
        html = ATTACHMENT_RE.sub(replace_attachment, source)
        
        # Add markdown="block" to div tags so that markdown content inside them gets processed
        html = re.sub(r'<div(\s+[^>]*)>', r'<div\1 markdown="block">', html)
        html = re.sub(r'<div>', r'<div markdown="block">', html)
        
        # Use markdown library for proper conversion
        md = markdown.Markdown(extensions=[
            TableExtension(),
            FencedCodeExtension(),
            'md_in_html',
        ])
        html = md.convert(html)

        # Notebook frontends style generated headings through :has(), because Markdown provides
        # no portable class syntax. Our renderer can add the class directly, which also gives PDF
        # pagination code a simple, broadly supported selector.
        html = re.sub(
            r'<h([23])>(?=<span class="question-heading-marker")',
            r'<h\1 class="question-heading">',
            html,
        )
        
        return html
        
    except ImportError:
        pass  # Fall through to basic conversion
    
    # Fallback: Basic markdown to HTML conversion (no table support)
    # Get cell-level attachments
    cell_attachments = cell.get('attachments', {})
    
    # Replace attachment references with data URIs
    def replace_attachment(match):
        alt = match.group(1)
        filename = match.group(2)
        
        # Check cell attachments first
        if filename in cell_attachments:
            data = cell_attachments[filename]
            for mime_type, b64data in data.items():
                return f'<img src="data:{mime_type};base64,{b64data}" alt="{alt}">'
        
        # Fall back to global attachments
        if filename in attachments:
            return f'<img src="{attachments[filename]}" alt="{alt}">'
        
        return f'<em>[Image not found: {filename}]</em>'
    
    html = ATTACHMENT_RE.sub(replace_attachment, source)
    
    # Convert markdown tables to HTML
    html = _convert_markdown_tables(html)
    
    # Headers
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    html = re.sub(
        r'<h([23])>(?=<span class="question-heading-marker")',
        r'<h\1 class="question-heading">',
        html,
    )

    # Bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)
    
    # Code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Lists (basic support)
    html = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^[-*] (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # Line breaks
    html = html.replace('\n\n', '</p><p>')
    html = f'<p>{html}</p>'
    
    # Handle regular markdown images (not attachments)
    # Supports both ![alt](url) and ![alt](<url with spaces>)
    html = re.sub(r'!\[([^\]]*)\]\(<?([^)>]+)>?\)', r'<img src="\2" alt="\1">', html)
    
    return html


def _convert_markdown_tables(text: str) -> str:
    """Convert markdown tables to HTML tables."""
    lines = text.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this looks like a table row (has | characters)
        if '|' in line and i + 1 < len(lines) and '|' in lines[i + 1]:
            # Check if next line is a separator (contains --)
            next_line = lines[i + 1]
            if re.match(r'^[\s|:-]+$', next_line) and '--' in next_line or '---' in next_line.replace(' ', ''):
                # This is a table! Parse it
                table_lines = [line]
                j = i + 1
                while j < len(lines) and '|' in lines[j]:
                    table_lines.append(lines[j])
                    j += 1
                
                # Convert table to HTML
                html_table = _parse_markdown_table(table_lines)
                result.append(html_table)
                i = j
                continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


def _parse_markdown_table(lines: List[str]) -> str:
    """Parse markdown table lines into HTML table."""
    if len(lines) < 2:
        return '\n'.join(lines)
    
    def parse_row(line: str) -> List[str]:
        # Split by | and strip whitespace, ignoring empty first/last elements
        cells = [c.strip() for c in line.split('|')]
        # Remove empty strings from start/end (from leading/trailing |)
        if cells and cells[0] == '':
            cells = cells[1:]
        if cells and cells[-1] == '':
            cells = cells[:-1]
        return cells
    
    header_cells = parse_row(lines[0])
    # Skip separator line (lines[1])
    
    html = ['<table style="border-collapse: collapse; margin: 10px 0;">']
    
    # Header row
    html.append('<thead><tr>')
    for cell in header_cells:
        html.append(f'<th style="border: 1px solid #ddd; padding: 8px; background: #f5f5f5;">{cell}</th>')
    html.append('</tr></thead>')
    
    # Body rows
    html.append('<tbody>')
    for line in lines[2:]:
        if not line.strip():
            continue
        cells = parse_row(line)
        html.append('<tr>')
        for cell in cells:
            html.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{cell}</td>')
        html.append('</tr>')
    html.append('</tbody>')
    
    html.append('</table>')
    return '\n'.join(html)


def _render_cell_outputs(cell: NotebookNode) -> str:
    """Render cell outputs, especially images, to HTML."""
    outputs = cell.get('outputs', [])
    if not outputs:
        return ''
    
    html_parts = []
    
    for output in outputs:
        output_type = output.get('output_type', '')
        
        if output_type == 'display_data' or output_type == 'execute_result':
            data = output.get('data', {})
            
            # Prefer PNG, then JPEG, then SVG
            if 'image/png' in data:
                b64 = data['image/png']
                html_parts.append(f'<img src="data:image/png;base64,{b64}">')
            elif 'image/jpeg' in data:
                b64 = data['image/jpeg']
                html_parts.append(f'<img src="data:image/jpeg;base64,{b64}">')
            elif 'image/svg+xml' in data:
                svg = data['image/svg+xml']
                html_parts.append(svg)
            elif 'text/html' in data:
                html_parts.append(data['text/html'])
            elif 'text/plain' in data:
                text = data['text/plain']
                html_parts.append(f'<div class="cell-output"><pre>{_escape_html(text)}</pre></div>')
        
        elif output_type == 'stream':
            text = output.get('text', '')
            if text.strip():
                html_parts.append(f'<div class="cell-output"><pre>{_escape_html(text)}</pre></div>')
    
    return '\n'.join(html_parts)


def _embed_local_images(html: str, base_path: Path) -> str:
    """Embed local images as base64 data URIs in HTML.
    
    Finds all <img> tags with src attributes that are local file paths
    (not data: URIs or URLs) and replaces them with embedded base64 data URIs.
    
    Args:
        html: HTML content with image references
        base_path: Base directory for resolving relative paths
    
    Returns:
        HTML with local images embedded as base64 data URIs
    
    Raises:
        FileNotFoundError: If a referenced image file does not exist
    """
    # Match <img src="..."> tags, capturing the src value
    img_pattern = re.compile(r'<img\s+([^>]*)src="([^"]+)"([^>]*)>', re.IGNORECASE)
    
    def replace_image(match):
        before_src = match.group(1)
        src = match.group(2)
        after_src = match.group(3)
        
        # Skip data URIs (already embedded) and http(s) URLs
        if src.startswith('data:') or src.startswith('http://') or src.startswith('https://'):
            return match.group(0)
        
        # Resolve the image path relative to base_path
        image_path = base_path / src
        
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {src}\n"
                f"  Looked in: {image_path}\n"
                f"  Make sure the image file exists and the path is correct."
            )
        
        # Determine MIME type from extension
        suffix = image_path.suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.webp': 'image/webp',
        }
        mime_type = mime_types.get(suffix, 'application/octet-stream')
        
        # Read and encode the image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('ascii')
        
        # Create data URI
        data_uri = f'data:{mime_type};base64,{image_data}'
        
        return f'<img {before_src}src="{data_uri}"{after_src}>'
    
    return img_pattern.sub(replace_image, html)


def html_to_pdf(html: str, output_path: Path, base_url: Path = None) -> bool:
    """Convert HTML to PDF using weasyprint.
    
    Args:
        html: HTML content to convert
        output_path: Path for the output PDF
        base_url: Base directory for resolving relative paths (e.g., Files/image.png)
    """
    WeasyprintHTML(string=html, base_url=str(base_url) if base_url else None).write_pdf(str(output_path))
    return True


_DEFAULT_PAGE_RULE = '@page { size: A4; margin: 12mm; } /* PSET_PAGE_SIZE */'
_A4_HEIGHT_MM = 297
_TALL_PAGE_STEP_MM = _A4_HEIGHT_MM / 4


def _render_question_pdf_pages(
    question_documents: List[SubmissionQuestionHtml],
    *,
    base_url: Optional[Path] = None,
) -> Tuple[list, list, List[Tuple[int, str, str]]]:
    """Render one variable-height WeasyPrint Page per logical question.

    Each question is first laid out on ordinary A4 pages. Longer questions are then re-rendered
    on one A4-width page, choosing the first quarter-A4 height that fits. Quarter-page steps avoid
    the large blank region caused by rounding any slight overflow up to a whole extra A4 sheet.
    """
    if not question_documents:
        raise ValueError("Cannot generate a submission PDF with no questions.")

    rendered_documents = []
    rendered_pages = []
    page_manifest = []
    base_url_string = str(base_url) if base_url else None

    for page_number, question in enumerate(question_documents, start=1):
        html = _embed_local_images(question.html, base_url or Path.cwd())
        natural_document = WeasyprintHTML(
            string=html, base_url=base_url_string).render()
        a4_equivalents = max(1, len(natural_document.pages))

        if a4_equivalents == 1:
            question_document = natural_document
        else:
            # A question that naturally occupies N A4 pages normally needs between N-1 and N
            # sheets of height once the intermediate page margins are removed. Try quarter-sheet
            # steps in that interval and retain the shortest height that produces one page.
            first_quarter = max(5, (a4_equivalents - 1) * 4 + 1)
            final_quarter = a4_equivalents * 4
            question_document = None
            height_mm = None
            for quarter_count in range(first_quarter, final_quarter + 1):
                candidate_height_mm = _TALL_PAGE_STEP_MM * quarter_count
                page_rule = (
                    f'@page {{ size: 210mm {candidate_height_mm}mm; margin: 12mm; }}'
                )
                tall_html = html.replace(_DEFAULT_PAGE_RULE, page_rule)
                if tall_html == html:
                    raise RuntimeError("Submission HTML is missing its replaceable @page rule.")
                candidate_document = WeasyprintHTML(
                    string=tall_html, base_url=base_url_string).render()
                if len(candidate_document.pages) == 1:
                    question_document = candidate_document
                    height_mm = candidate_height_mm
                    break

            # The N-sheet upper bound should fit because the tall page has only one pair of
            # margins. Retain the former defensive growth path for unusual future layouts.
            if question_document is None:
                height_mm = _A4_HEIGHT_MM * a4_equivalents
                for _attempt in range(4):
                    page_rule = f'@page {{ size: 210mm {height_mm}mm; margin: 12mm; }}'
                    tall_html = html.replace(_DEFAULT_PAGE_RULE, page_rule)
                    question_document = WeasyprintHTML(
                        string=tall_html, base_url=base_url_string).render()
                    if len(question_document.pages) == 1:
                        break
                    height_mm += _A4_HEIGHT_MM * (len(question_document.pages) - 1)
                else:
                    raise RuntimeError(
                        f"Question {question.name} could not be rendered onto one tall PDF page."
                    )

        if len(question_document.pages) != 1:
            raise RuntimeError(
                f"Question {question.name} produced {len(question_document.pages)} PDF pages; "
                "expected exactly one."
            )
        rendered_documents.append(question_document)  # Keep font/image resources alive.
        rendered_pages.append(question_document.pages[0])
        page_manifest.append((page_number, question.name, question.question_type))
        if question_document.pages[0].height >= natural_document.pages[0].height * 4:
            print(
                f"⚠️  Question {question.name} occupies one page that is "
                "at least 4 A4 sheets tall. Check for accidental excessive output."
            )

    return rendered_documents, rendered_pages, page_manifest


def write_question_paged_pdf(
    question_documents: List[SubmissionQuestionHtml],
    output_path: Path,
    *,
    base_url: Optional[Path] = None,
) -> List[Tuple[int, str, str]]:
    """Write exactly one variable-height physical PDF page per logical question.

    Returns ``(page_number, question_name, question_type)`` entries for validation and logging.
    """
    rendered_documents, rendered_pages, page_manifest = _render_question_pdf_pages(
        question_documents, base_url=base_url)
    # Document.copy explicitly supports combining Page objects from multiple rendered documents,
    # preserving vector text and images without another PDF dependency.
    rendered_documents[0].copy(rendered_pages).write_pdf(str(output_path))
    return page_manifest


# -----------------------------
# Main function
# -----------------------------

def _resolve_notebook_path(explicit: Optional[str]) -> Path:
    """Work out which notebook to submit, without depending on its filename.

    A hardcoded filename in the generated submission cell is fragile twice over: it goes stale if
    the build fails to rewrite it, and it breaks outright the moment a student renames their
    notebook. Resolving instead means neither can happen.

    Order:
      1. An explicit argument, when given -- unchanged behaviour, and the escape hatch when a
         folder genuinely holds several notebooks.
      2. ``__vsc_ipynb_file__``, which VS Code's Jupyter extension defines in notebook globals.
         This names the exact executing notebook even with several side by side. Best-effort:
         other front-ends do not set it.
      3. The single ``*.ipynb`` beside this module's parent (Files/ lives inside the notebook's
         folder, so the parent IS the notebook's folder).

    Anything else -- no candidates, or more than one with no VS Code hint -- raises. Guessing
    would risk submitting the wrong work silently, which is worse than an error.
    """
    if explicit is not None:
        return Path(explicit).resolve()

    # VS Code sets this global in the *notebook's* namespace, so look up the caller's frame chain
    # rather than this module's globals.
    frame = inspect.currentframe()
    while frame is not None:
        candidate = frame.f_globals.get('__vsc_ipynb_file__')
        if candidate:
            resolved = Path(candidate).resolve()
            if resolved.exists():
                return resolved
        frame = frame.f_back

    search_dir = Path(__file__).resolve().parent.parent
    candidates = sorted(p for p in search_dir.glob('*.ipynb') if not p.name.startswith('.'))

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No notebook found in {search_dir}.\n"
            "Run this from the folder holding your assignment notebook, or name it explicitly:\n"
            "    generate_pset_submission('my_notebook.ipynb')"
        )

    listing = "\n".join(f"      {p.name}" for p in candidates)
    raise ValueError(
        f"Found {len(candidates)} notebooks in {search_dir}, so I cannot tell which one to "
        f"submit:\n{listing}\n"
        "Name the one you want:\n"
        "    generate_pset_submission('my_notebook.ipynb')"
    )


def _resolve_student_email(explicit: Optional[str]) -> str:
    """Resolve ``my_email`` from an explicit argument or the calling notebook's globals."""
    if explicit is not None:
        return explicit

    frame = inspect.currentframe()
    try:
        while frame is not None:
            if "my_email" in frame.f_globals:
                return frame.f_globals["my_email"]
            frame = frame.f_back
    finally:
        # Avoid retaining a frame reference (and everything reachable from it).
        del frame

    raise RuntimeError(
        "my_email is not defined. Run the identity cell near the top of the notebook, "
        "then generate the submission again."
    )


# Kept in sync with notebook_types.PSET_ASSIGNMENT_ID_METADATA_KEY. notebook_types.py is not
# shipped into Files/, so importing the constant here would break released notebooks.
PSET_ASSIGNMENT_ID_METADATA_KEY = "pset_assignment_id"
_PSET_ASSIGNMENT_ID_RE = re.compile(r"^PSET[_ -]*(\d+)$", re.IGNORECASE)
_UNSAFE_FILENAME_CHARACTER_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _submission_assignment_label(notebook: NotebookNode) -> str:
    """Return the filename label stamped into the notebook by the release builder.

    ``PSET_1`` becomes the more readable ``PSET1``. Missing metadata is an error: inferring the
    assignment from a student-editable filename could silently put work in the wrong folder.
    """
    assignment_id = notebook.get("metadata", {}).get(PSET_ASSIGNMENT_ID_METADATA_KEY)
    if not isinstance(assignment_id, str) or not assignment_id.strip():
        raise RuntimeError(
            f"This notebook is missing its {PSET_ASSIGNMENT_ID_METADATA_KEY!r} assignment "
            "metadata. Rebuild the assignment from its master instructor notebook or download "
            "a current released notebook before generating a submission."
        )

    assignment_id = assignment_id.strip()
    match = _PSET_ASSIGNMENT_ID_RE.fullmatch(assignment_id)
    label = f"PSET{match.group(1)}" if match else assignment_id
    if _UNSAFE_FILENAME_CHARACTER_RE.search(label) or label.endswith((" ", ".")):
        raise ValueError(
            f"Assignment id {assignment_id!r} cannot be used safely in a folder name. "
            "Rename the master notebook with a filesystem-safe assignment name and rebuild."
        )
    return label


def submission_directory_name(notebook: NotebookNode) -> str:
    """Name of the assignment-specific folder holding generated submission files."""
    return f"{_submission_assignment_label(notebook)}_to_submit"


def submission_pdf_name(normalized_email: str) -> str:
    """Name the answers PDF with the part of the Gradescope email before ``@``."""
    if (
        not isinstance(normalized_email, str)
        or normalized_email.count("@") != 1
        or not all(normalized_email.split("@", 1))
    ):
        raise ValueError("A normalized Gradescope email is required to name the answers PDF.")

    email_id = normalized_email.split("@", 1)[0]
    if _UNSAFE_FILENAME_CHARACTER_RE.search(email_id) or email_id.endswith((" ", ".")):
        raise ValueError(
            f"The email ID from Gradescope email {normalized_email!r} contains a character "
            "that cannot be used safely in a filename."
        )
    return f"answers_{email_id}.pdf"


def generate_pset_submission(notebook_path: Optional[str] = None,
                             allow_unchecked: bool = False,
                             student_email: Optional[str] = None) -> Tuple[Path, Path]:
    """
    Generate Gradescope submission files from a completed assignment notebook.

    Creates an assignment-specific folder such as ``PSET1_to_submit`` next to the notebook. It
    contains code.py, answers_<email-id>.pdf, and (for assignments with in-notebook checks)
    signed_results.json.

    Parameters
    ----------
    notebook_path : str, optional
        Path to the student's completed notebook (.ipynb). Omit it and the notebook is found
        automatically (see _resolve_notebook_path) -- which keeps working if you rename the file.
        Pass it explicitly only when the folder holds several notebooks.
    allow_unchecked : bool
        If any auto-graded question has not been checked with check_answer(), abort by
        default. Pass allow_unchecked=True to submit partial work anyway; unchecked
        questions are scored 0.
    student_email : str, optional
        Email address associated with the student's Gradescope account. In a notebook this
        defaults to the generated ``my_email`` variable; command-line and build callers should
        pass it explicitly.

    Returns
    -------
    tuple of (code_path, pdf_path)
        Paths to the generated .py and .pdf files

    Example
    -------
    >>> from generate_pset_submission import generate_pset_submission
    >>> generate_pset_submission()
    Generated: PSET1_to_submit/code.py
    Generated: PSET1_to_submit/answers_student.pdf
    """
    nb_path = _resolve_notebook_path(notebook_path)

    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    if not nb_path.suffix == '.ipynb':
        raise ValueError("Input must be a Jupyter notebook (.ipynb)")

    # Load notebook
    print(f"📖 Loading notebook: {nb_path.name}")
    nb = load_notebook(nb_path)

    # Submission naming is carried by release metadata, not inferred from the notebook filename.
    # Resolve and validate both identifiers before creating anything on disk.
    out_dir = nb_path.parent / submission_directory_name(nb)
    code_path = out_dir / 'code.py'
    try:
        normalized_email = signing.normalize_email(_resolve_student_email(student_email))
        pdf_path = out_dir / submission_pdf_name(normalized_email)
    except (RuntimeError, ValueError) as identity_error:
        print(f"\n⛔ {identity_error}")
        print("Submission aborted — no files were written.")
        # Preserve the historical two-Path return contract for notebook Run All. This sentinel is
        # never created and says exactly why there is no real PDF path yet.
        return code_path, out_dir / "answers_EMAIL_NOT_SET.pdf"

    # Find questions
    questions = find_question_cells(nb)
    
    if not questions:
        print("⚠️  No QUESTION directives found in notebook.")
        print("   Make sure your notebook uses <!-- QUESTION name POINTS n TYPE auto|manual --> markers.")
        return code_path, pdf_path
    
    auto_qs = [q for q in questions if q[3] == 'auto']
    manual_qs = [q for q in questions if q[3] == 'manual']

    print(f"   Found {len(auto_qs)} auto-graded and {len(manual_qs)} manually-graded questions")

    # Build the signed results from the recorded check_answer() log. This also enforces that
    # every auto-graded question was checked (unless allow_unchecked=True). Assignments with
    # no in-notebook checks (no blob) simply skip this.
    signed_results = None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import pset_check
        signed_results = pset_check.build_signed_results(
            normalized_email, allow_unchecked=allow_unchecked
        )
    except ImportError:
        # No checker module beside us (e.g. a Files/ that ships only the submission script).
        # Degrade to "nothing to sign" rather than crashing with a ModuleNotFoundError.
        print("ℹ️  No in-notebook checker found; skipping signed_results.json.")
    except FileNotFoundError:
        pass  # No answer-check blob in this assignment; nothing to sign.
    except (RuntimeError, ValueError) as check_error:
        print(f"\n⛔ {check_error}")
        print("Submission aborted — no files were written.")
        return code_path, pdf_path

    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate code file for autograding
    if auto_qs:
        code_content = extract_code_for_autograding(nb, questions)
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        print(f"✅ Generated: {code_path.name}")
    else:
        print("ℹ️  No auto-graded questions found, skipping code file.")
    
    # Write the signed results file (grade of record for the auto-graded questions).
    if signed_results is not None:
        signed_path = out_dir / 'signed_results.json'
        signed_path.write_text(json.dumps(signed_results, indent=2), encoding='utf-8')
        print(f"✅ Generated: {signed_path.name}")

    # Generate the manual-grading PDF and auto-graded audit trail. ``questions`` is nonempty here,
    # and every declared question receives a page even if it has no exported response content.
    _math_render_failures.clear()
    question_documents = extract_manual_content_html(
        nb, questions, nb_path, split_questions=True)
    page_manifest = write_question_paged_pdf(
        question_documents, pdf_path, base_url=nb_path.parent)
    print(f"✅ Generated: {pdf_path.name} ({len(page_manifest)} question pages)")
    # Report every LaTeX fragment that would not render. These are marked in red in the PDF
    # itself; the summary is here so they cannot be missed by scrolling past them.
    if _math_render_failures:
        print(f"\n⚠️  {len(_math_render_failures)} math expression(s) could not be rendered "
              f"and appear as red placeholders in {pdf_path.name}:")
        for failure in _math_render_failures:
            print(f"   - {failure}")
        print("   Fix the LaTeX and re-run generate_pset_submission() before submitting.")
    
    # Copy the full notebook for debugging/reference
    notebook_copy_path = out_dir / nb_path.name
    shutil.copy2(nb_path, notebook_copy_path)
    print(f"✅ Copied: {nb_path.name}")
    
    print()
    print(f"📦 Submission files ready in {out_dir.name}/ folder!")
    print(f"   Upload the files from {out_dir.name} to Gradescope:")
    if signed_results is not None:
        print(f"   • signed_results.json → Autograder assignment (your recorded scores)")
    if auto_qs:
        print(f"   • code.py → submission record")
    print(f"   • {pdf_path.name} → Manual grading assignment")
    print(f"   • {nb_path.name} → Full notebook (for reference/debugging)")
    print(f"\n✅ Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show "View PDF" button if in interactive notebook context and PDF was generated
    if pdf_path.exists():
        _show_view_pdf_button(pdf_path)
    
    return code_path, pdf_path


def _show_view_pdf_button(pdf_path: Path) -> None:
    """Display a button to open the PDF in the system viewer."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
        
        # Check if we're in an interactive context
        try:
            get_ipython()  # type: ignore
        except NameError:
            return  # Not in IPython/Jupyter
        
        def on_click(button):
            # Open PDF in system default viewer
            webbrowser.open(f'file://{pdf_path}')
        
        button = widgets.Button(
            description='📄 View PDF',
            button_style='info',
            tooltip=f'Open {pdf_path.name} in your default PDF viewer',
            layout=widgets.Layout(width='120px', margin='10px 0')
        )
        button.on_click(on_click)
        display(button)
    except ImportError:
        # ipywidgets not available, skip button
        pass


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate Gradescope submission files from a Jupyter notebook.'
    )
    parser.add_argument(
        'notebook', nargs='?', default=None,
        help='Path to the completed assignment notebook (.ipynb). Omit to auto-detect the '
             'single notebook in the assignment folder.'
    )
    parser.add_argument(
        '--allow-unchecked', action='store_true',
        help='Submit even if some auto-graded questions were never checked (they score 0).'
    )
    parser.add_argument(
        '--student-email',
        help='Email address associated with your Gradescope account (required for assignments '
             'with in-notebook checks).'
    )

    args = parser.parse_args()

    try:
        generate_pset_submission(
            args.notebook,
            allow_unchecked=args.allow_unchecked,
            student_email=args.student_email,
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
