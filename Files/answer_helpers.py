"""answer_helpers.py

Helper functions for students to add images and text answers to their notebook answers.

Usage:
    from answer_helpers import add_image, add_answer, add_short_answers, add_table_answer
    add_image("q7_1")          # Buttons for file selection or clipboard paste
    add_answer("q7_2")         # File / clipboard / typed-text-with-math for one answer
    add_short_answers("1")     # One widget: a labeled text box per @@SHORT_ANSWER prompt of block "1"
                               # (prompts loaded from Files/short_answer_prompts.json)
    add_table_answer("rates")  # Structured choice/text table from Files/table_answer_schemas.json
"""

from pathlib import Path
from io import BytesIO
from datetime import datetime
import json
import shutil

import ipywidgets as widgets
from IPython.display import display, Markdown
from PIL import Image, ImageGrab, ImageOps

try:
    from .answer_pages import AnswerPage, PAGE_MANIFEST_SUFFIX, read_answer_response
except ImportError:
    from answer_pages import AnswerPage, PAGE_MANIFEST_SUFFIX, read_answer_response


def _answer_root() -> Path:
    """Return the folder that accompanies the notebook containing this shipped helper.

    Release builds place this module in ``<notebook folder>/Files``. Anchoring from ``__file__``
    keeps answer saves beside the notebook even when a front-end starts its kernel with a
    different working directory. When the development copy in ``psetTools`` is imported directly,
    retain the convenient historical behavior of using the current working directory.
    """
    module_dir = Path(__file__).resolve().parent
    if module_dir.name == "Files":
        return module_dir.parent
    return Path.cwd()


ANSWER_ROOT = _answer_root()
ANSWERS_DIR = ANSWER_ROOT / "Answers"
LEGACY_ANSWERS_DIR = ANSWER_ROOT / "Images"
def _existing_answer_path(answer_id: str, suffix: str) -> Path:
    """Prefer a new ``Answers`` file, falling back read-only to the old ``Images`` location."""
    current = ANSWERS_DIR / f"{answer_id}{suffix}"
    if current.exists():
        return current
    return LEGACY_ANSWERS_DIR / f"{answer_id}{suffix}"


# Visible rows for a short-answer box when its prompt carries no LINES value. Mirrors
# notebook_types.SHORT_ANSWER_DEFAULT_LINES, duplicated here because this module ships into the
# student's Files/ folder without the build-time modules.
DEFAULT_ANSWER_LINES = 4

# Maximum display width in pixels for preview (matches 6.5" at ~96 DPI screen resolution)
MAX_DISPLAY_WIDTH = 624


DEFAULT_ANSWER_FORMATS = ("image", "text")
VALID_ANSWER_FORMATS = frozenset(DEFAULT_ANSWER_FORMATS)


def _normalize_answer_formats(allowed_formats) -> tuple:
    """Validate add_answer()'s UI-format selection while preserving its display order."""
    if isinstance(allowed_formats, str):
        formats = (allowed_formats.lower(),)
    else:
        try:
            formats = tuple(str(value).lower() for value in allowed_formats)
        except TypeError as exc:
            raise ValueError("allowed_formats must contain image, text, or both.") from exc
    if not formats:
        raise ValueError("allowed_formats must contain at least one answer format.")
    if len(set(formats)) != len(formats):
        raise ValueError("allowed_formats must not repeat a format.")
    unknown = sorted(set(formats) - VALID_ANSWER_FORMATS)
    if unknown:
        raise ValueError(
            "Unknown answer format(s): "
            + ", ".join(unknown)
            + ". Expected image, text, or both."
        )
    return formats


def display_image_with_width(image_path: Path, max_width: int = MAX_DISPLAY_WIDTH) -> None:
    """Display an image with explicit HTML width (VS Code doesn't honor IPImage width)."""
    import base64
    with Image.open(image_path) as img:
        display_width = min(img.width, max_width)
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    display(widgets.HTML(
        f'<img src="data:image/png;base64,{image_data}" '
        f'style="max-width: {display_width}px; width: 100%;" />'
    ))


def save_typed_answer(content: str, text_path: Path, image_path: Path) -> None:
    """Persist a typed answer as markdown source (with frontmatter) at Answers/{id}.md.

    Used by add_short_answers()'s per-box save. The markdown is the single source of truth:
    the submission PDF renders it through the same markdown+math pipeline as every other cell,
    giving real word wrap,
    real bold/lists, and vector math. Typed answers used to be rasterized to a PNG here, which
    lost all three.

    Any PNG left over from that older behaviour is removed, because a stale image would otherwise
    shadow the freshly typed text when the PDF is built.
    """
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    md_content = f"""---
answer_type: text
created: {timestamp}
---
{content}
"""
    text_path.write_text(md_content, encoding='utf-8')
    if image_path.exists():
        image_path.unlink()


def add_image(image_id: str) -> None:
    """
    Display buttons for adding an image to an answer.
    
    Creates two buttons:
    - "Select Image File" - opens file picker dialog
    - "Paste from Clipboard" - grabs image from system clipboard
    
    The image is saved to Answers/{image_id}.png and displayed.
    If an image already exists, it is displayed immediately.
    
    Parameters
    ----------
    image_id : str
        Identifier for the image, e.g., "q7_1" for question 7.1.
        The file will be saved as Answers/{image_id}.png
    """
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = ANSWERS_DIR / f"{image_id}.png"
    
    # Output area for displaying the image and status messages
    output = widgets.Output()
    
    def save_and_display_image(pil_image: Image.Image, source: str) -> None:
        """Save PIL image as PNG and display it."""
        # Convert to RGB if necessary (handles RGBA, palette modes, etc.)
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparency
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGBA')
            background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
            pil_image = background
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Save as PNG
        pil_image.save(image_path, 'PNG')
        
        # Display the image
        with output:
            output.clear_output(wait=True)
            print(f"✓ Image saved from {source}: {image_path}")
            display_image_with_width(image_path)
    
    def show_existing_image() -> None:
        """Display the existing image if present."""
        existing_image_path = _existing_answer_path(image_id, ".png")
        if existing_image_path.exists():
            with output:
                output.clear_output(wait=True)
                print(f"Current image: {existing_image_path}")
                display_image_with_width(existing_image_path)
    
    # --- File Upload Widget ---
    file_upload = widgets.FileUpload(
        accept='image/*',
        multiple=False,
        description='Select Image File',
        button_style='primary',
        layout=widgets.Layout(width='180px')
    )
    
    def on_file_upload(change):
        """Handle file upload."""
        if not file_upload.value:
            return
        
        try:
            # Get the uploaded file data
            # ipywidgets 8.x: value is a tuple of dicts with 'content' as bytes
            # ipywidgets 7.x: value is a dict of dicts with 'content' as bytes
            uploaded = file_upload.value
            if isinstance(uploaded, tuple):
                # ipywidgets 8.x format
                file_info = uploaded[0]
                image_data = file_info['content']
            else:
                # ipywidgets 7.x format
                file_info = list(uploaded.values())[0]
                image_data = file_info['content']
            
            # Open with PIL and save as PNG
            pil_image = Image.open(BytesIO(image_data))
            save_and_display_image(pil_image, "file")
            
        except Exception as error:
            with output:
                output.clear_output(wait=True)
                print(f"✗ Error loading image: {error}")
    
    file_upload.observe(on_file_upload, names='value')
    
    # --- Paste Button ---
    paste_button = widgets.Button(
        description='Paste from Clipboard',
        button_style='info',
        icon='clipboard',
        layout=widgets.Layout(width='180px')
    )
    
    def on_paste_click(button):
        """Handle paste button click."""
        try:
            # Grab image from clipboard
            clipboard_image = ImageGrab.grabclipboard()
            
            if clipboard_image is None:
                with output:
                    output.clear_output(wait=True)
                    print("✗ No image found in clipboard.")
                    print("  Copy an image first, then click 'Paste from Clipboard'.")
                return
            
            if not isinstance(clipboard_image, Image.Image):
                # On macOS, might get a list of file paths instead
                if isinstance(clipboard_image, list) and len(clipboard_image) > 0:
                    # Try to open as file path
                    clipboard_image = Image.open(clipboard_image[0])
                else:
                    with output:
                        output.clear_output(wait=True)
                        print("✗ Clipboard does not contain an image.")
                    return
            
            save_and_display_image(clipboard_image, "clipboard")
            
        except Exception as error:
            with output:
                output.clear_output(wait=True)
                print(f"✗ Error pasting image: {error}")
    
    paste_button.on_click(on_paste_click)
    
    # --- Layout ---
    button_row = widgets.HBox(
        [file_upload, paste_button],
        layout=widgets.Layout(gap='10px')
    )
    
    label = widgets.HTML(
        value=f"<b>Image:</b> <code>{image_id}</code> → <code>{image_path}</code>"
    )
    
    container = widgets.VBox([label, button_row, output])
    display(container)
    
    # Show existing image if present
    show_existing_image()


def add_answer(answer_id: str, allowed_formats=DEFAULT_ANSWER_FORMATS) -> None:
    """Edit one ordered answer containing text pages, image pages, or both."""
    allowed_formats = _normalize_answer_formats(allowed_formats)
    image_allowed = "image" in allowed_formats
    text_allowed = "text" in allowed_formats
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ANSWERS_DIR / f"{answer_id}{PAGE_MANIFEST_SUFFIX}"
    pages = list(read_answer_response(ANSWERS_DIR.parent, answer_id).pages)
    editing_index = None

    page_output = widgets.Output()
    editor_output = widgets.Output()
    text_area = widgets.Textarea(
        placeholder='Type this page here. Markdown and $LaTeX$ math are supported.',
        layout=widgets.Layout(width='100%', height='200px'),
    )
    editor = widgets.VBox(layout=widgets.Layout(display='none'))

    def next_page_path(kind: str) -> Path:
        index = 1
        while (ANSWERS_DIR / f"{answer_id}_page_{index:03d}.md").exists() or (
            ANSWERS_DIR / f"{answer_id}_page_{index:03d}.png"
        ).exists():
            index += 1
        suffix = '.md' if kind == 'text' else '.png'
        return ANSWERS_DIR / f"{answer_id}_page_{index:03d}{suffix}"

    def save_manifest() -> None:
        payload = {
            'schema_version': 2,
            'pages': [{'type': page.kind, 'file': page.path.name} for page in pages],
        }
        temporary = manifest.with_suffix(manifest.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        temporary.replace(manifest)

    def migrate_if_needed() -> None:
        """Copy old responses before changing their order; keep Images/ read-only."""
        nonlocal pages
        if manifest.exists():
            return
        old_pages = pages
        pages = []
        for old in old_pages:
            destination = next_page_path(old.kind)
            shutil.copy2(old.path, destination)
            pages.append(AnswerPage(old.kind, destination))
        save_manifest()
        for old in old_pages:
            if old.path.parent == ANSWERS_DIR:
                old.path.unlink(missing_ok=True)
        (ANSWERS_DIR / f"{answer_id}.images.json").unlink(missing_ok=True)
        (ANSWERS_DIR / f"{answer_id}.md").unlink(missing_ok=True)
        (ANSWERS_DIR / f"{answer_id}.png").unlink(missing_ok=True)

    def refresh(status: str | None = None) -> None:
        with page_output:
            page_output.clear_output(wait=True)
            if status:
                display(widgets.HTML(f'<b>{status}</b>'))
            if not pages:
                print('No pages saved yet.')
                return
            for index, page in enumerate(pages):
                up = widgets.Button(description='↑', tooltip='Move page up',
                                    layout=widgets.Layout(width='42px'))
                down = widgets.Button(description='↓', tooltip='Move page down',
                                      layout=widgets.Layout(width='42px'))
                delete = widgets.Button(description='Delete', button_style='danger',
                                        layout=widgets.Layout(width='80px'))
                up.disabled = index == 0
                down.disabled = index == len(pages) - 1
                up.on_click(lambda _button, i=index: move_page(i, -1))
                down.on_click(lambda _button, i=index: move_page(i, 1))
                delete.on_click(lambda _button, i=index: delete_page(i))
                controls = [widgets.HTML(f'<b>Page {index + 1} ({page.kind})</b>'), up, down]
                if page.kind == 'text' and text_allowed:
                    edit = widgets.Button(description='Edit', layout=widgets.Layout(width='70px'))
                    edit.on_click(lambda _button, i=index: open_editor(i))
                    controls.append(edit)
                controls.append(delete)
                preview = widgets.Output()
                with preview:
                    if page.kind == 'text':
                        content = page.path.read_text(encoding='utf-8')
                        if content.startswith('---'):
                            parts = content.split('---', 2)
                            if len(parts) >= 3:
                                content = parts[2].strip()
                        display(widgets.HTML('<span style="color: #28a745;">✓ Answer saved</span>'))
                        display(Markdown(content))
                    else:
                        display_image_with_width(page.path)
                display(widgets.VBox([
                    widgets.HBox(controls),
                    preview,
                ], layout=widgets.Layout(border='1px solid #ddd', padding='8px', margin='6px 0')))

    def move_page(index: int, offset: int) -> None:
        nonlocal editing_index
        migrate_if_needed()
        other = index + offset
        if not (0 <= index < len(pages) and 0 <= other < len(pages)):
            return
        pages[index], pages[other] = pages[other], pages[index]
        if editing_index == index:
            editing_index = other
        elif editing_index == other:
            editing_index = index
        save_manifest()
        refresh('✓ Page order saved.')

    def delete_page(index: int) -> None:
        nonlocal editing_index
        migrate_if_needed()
        if not 0 <= index < len(pages):
            return
        removed = pages.pop(index)
        if editing_index == index:
            editing_index = None  # Saving the still-open editor now adds a new page.
        elif editing_index is not None and editing_index > index:
            editing_index -= 1
        save_manifest()
        removed.path.unlink(missing_ok=True)
        refresh('✓ Page deleted.')

    def open_editor(index=None) -> None:
        nonlocal editing_index
        editing_index = index
        if index is None:
            text_area.value = ''
        else:
            content = pages[index].path.read_text(encoding='utf-8')
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    content = parts[2].strip()
            text_area.value = content
        editor.layout.display = 'block'
        with editor_output:
            editor_output.clear_output(wait=True)
            print('Editing page' if index is not None else 'Adding a text page')

    def save_text(_button) -> None:
        nonlocal editing_index
        content = text_area.value.strip()
        if not content:
            with editor_output:
                editor_output.clear_output(wait=True)
                print('Enter some text before saving this page.')
            return
        migrate_if_needed()
        if editing_index is None:
            path = next_page_path('text')
        else:
            path = pages[editing_index].path
        timestamp = datetime.now().isoformat()
        payload = f'---\nanswer_type: text\ncreated: {timestamp}\n---\n{content}\n'
        temporary = path.with_suffix('.md.tmp')
        temporary.write_text(payload, encoding='utf-8')
        temporary.replace(path)
        if editing_index is None:
            pages.append(AnswerPage('text', path))
            save_manifest()
        editing_index = None
        editor.layout.display = 'none'
        refresh('✓ Answer saved.')

    save_button = widgets.Button(description='Save & Preview', button_style='success')
    save_button.on_click(save_text)
    cancel_button = widgets.Button(description='Cancel')
    cancel_button.on_click(lambda _button: setattr(editor.layout, 'display', 'none'))
    editor.children = [text_area, widgets.HBox([save_button, cancel_button]), editor_output]

    def normalize_image(pil_image: Image.Image) -> Image.Image:
        pil_image = ImageOps.exif_transpose(pil_image)
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode != 'RGBA':
                pil_image = pil_image.convert('RGBA')
            background.paste(pil_image, mask=pil_image.getchannel('A'))
            return background
        return pil_image.convert('RGB')

    def append_image(pil_image: Image.Image, source: str) -> None:
        migrate_if_needed()
        path = next_page_path('image')
        normalize_image(pil_image).save(path, 'PNG')
        pages.append(AnswerPage('image', path))
        save_manifest()
        refresh(f'✓ Added page {len(pages)} from {source}.')

    file_upload = widgets.FileUpload(accept='image/*', multiple=True,
                                     description='Upload Image(s)', button_style='primary',
                                     layout=widgets.Layout(width='140px'))

    def on_file_upload(_change) -> None:
        if not file_upload.value:
            return
        try:
            uploaded = file_upload.value
            infos = list(uploaded) if isinstance(uploaded, (tuple, list)) else list(uploaded.values())
            for info in infos:
                with Image.open(BytesIO(info['content'])) as image:
                    append_image(image, info.get('name', 'file'))
        except Exception as exc:
            with page_output:
                page_output.clear_output(wait=True)
                print(f'✗ Error loading image: {exc}')

    file_upload.observe(on_file_upload, names='value')
    paste_button = widgets.Button(description='Paste Page', button_style='info', icon='clipboard',
                                  layout=widgets.Layout(width='140px'))

    def paste_image(_button) -> None:
        try:
            image = ImageGrab.grabclipboard()
            if isinstance(image, list) and image:
                image = Image.open(image[0])
            if not isinstance(image, Image.Image):
                raise ValueError('No image found in the clipboard.')
            append_image(image, 'clipboard')
        except Exception as exc:
            with page_output:
                page_output.clear_output(wait=True)
                print(f'✗ Error pasting image: {exc}')

    paste_button.on_click(paste_image)
    text_button = widgets.Button(description='Type Answer', button_style='warning', icon='pencil',
                                 layout=widgets.Layout(width='140px'))
    text_button.on_click(lambda _button: open_editor())
    clear_button = widgets.Button(description='Clear Answer', button_style='danger', icon='trash',
                                  layout=widgets.Layout(width='140px'))

    def clear_answer(_button) -> None:
        nonlocal pages, editing_index
        migrate_if_needed()
        old_pages = pages
        pages = []
        save_manifest()  # Empty manifest hides read-only legacy files in Images/.
        for page in old_pages:
            page.path.unlink(missing_ok=True)
        for suffix in ('.md', '.png'):
            for orphan in ANSWERS_DIR.glob(f'{answer_id}_page_*{suffix}'):
                orphan.unlink(missing_ok=True)
        editing_index = None
        text_area.value = ''
        editor.layout.display = 'none'
        refresh('Answer cleared.')

    clear_button.on_click(clear_answer)
    buttons = []
    if image_allowed:
        buttons.extend([file_upload, paste_button])
    if text_allowed:
        buttons.append(text_button)
    buttons.append(clear_button)
    children = [widgets.HTML(value=f'<b>Answer:</b> <code>{answer_id}</code>'),
                widgets.HBox(buttons),
                page_output]
    if text_allowed:
        children.append(editor)
    container = widgets.VBox(children,
                             layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='5px 0'))
    display(container)
    refresh()


def add_short_answers(block_key: str) -> None:
    """Render one labeled text box per prompt for a @@SHORT_ANSWER block.

    The prompt definitions (question text, ids, letters, points, box height) are loaded from the
    generated Files/short_answer_prompts.json shipped beside this module, keyed by the subpart
    display (e.g. "1" or "2.3"). Each answer saves to Answers/{id}.md, which the submission
    renderer still reads as a single text response.

    Boxes autosave when they lose focus, so an answer survives closing the notebook without
    pressing "Save all answers"; that button remains as an explicit save-everything backstop.
    """
    import json

    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(__file__).parent / "short_answer_prompts.json"
    try:
        all_prompts = json.loads(prompts_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        display(widgets.HTML(
            f'<b style="color:#cc0000;">short_answer_prompts.json not found at {prompts_path}</b>'))
        return

    prompts = all_prompts.get(str(block_key))
    if not prompts:
        display(widgets.HTML(
            f'<b style="color:#cc0000;">No short-answer prompts found for block "{block_key}".</b>'))
        return

    # Theme-aware colours. JupyterLab and the VS Code notebook renderer both define these CSS
    # variables, so the widget follows the editor into dark mode instead of forcing a light box.
    border_color = 'var(--jp-border-color1, #dddddd)'
    muted_color = 'var(--jp-ui-font-color2, #666666)'
    saved_color = '#28a745'

    rows = []
    entries = []  # one AnswerBox-ish tuple per prompt

    for prompt in prompts:
        prompt_id = prompt["id"]
        text_path = ANSWERS_DIR / f"{prompt_id}.md"
        image_path = ANSWERS_DIR / f"{prompt_id}.png"
        existing_text_path = _existing_answer_path(prompt_id, ".md")

        points = prompt.get("points")
        if points is None:
            points_chip = ''
        else:
            points_chip = (f'<span style="color:{muted_color}; font-size:0.85em; '
                           f'white-space:nowrap; padding-left:12px;">'
                           f'{points} {"pt" if points == 1 else "pts"}</span>')

        # The question is markdown and may contain $math$, so it must be rendered by the notebook
        # front-end -- display(Markdown(...)) inside an Output. NOT widgets.HTML/HTMLMath: those
        # take raw HTML, so "**bold**" would show its asterisks, and HTMLMath's MathJax pass does
        # not fire reliably in the VS Code notebook renderer.
        #
        # Only the (a)/(b) label is emphasised. Wrapping the whole question in ** would INVERT the
        # emphasis of any question that contains its own **bold** (markdown closes the outer span
        # at the inner marker), which is exactly what used to happen here and in the PDF.
        label_output = widgets.Output(layout=widgets.Layout(flex='1 1 auto'))
        with label_output:
            display(Markdown(f"**{prompt.get('label', '')}** {prompt.get('question', '')}"))

        label = widgets.HBox(
            [label_output, widgets.HTML(value=points_chip,
                                        layout=widgets.Layout(flex='0 0 auto'))],
            layout=widgets.Layout(width='100%', justify_content='space-between',
                                  align_items='flex-start'),
        )

        text_area = widgets.Textarea(
            value='',
            placeholder='Type your answer here...  ($x^2$ inline, $$E=mc^2$$ block math, **markdown** supported)',
            # continuous_update=False makes the value sync to Python only when the box loses
            # focus -- that is the hook the autosave below rides on.
            continuous_update=False,
            rows=int(prompt.get("lines") or DEFAULT_ANSWER_LINES),
            layout=widgets.Layout(width='99%'),
        )
        status_label = widgets.HTML(value='')
        preview_output = widgets.Output()

        # A left accent rule instead of a full border: far quieter with many prompts stacked, and
        # its colour doubles as the saved/unsaved indicator.
        row = widgets.VBox(
            [label, text_area, status_label, preview_output],
            layout=widgets.Layout(
                border_left=f'3px solid {border_color}',
                padding='4px 0 4px 12px',
                margin='10px 0',
            ),
        )
        rows.append(row)
        entries.append({
            'text_area': text_area,
            'text_path': text_path,
            'image_path': image_path,
            'existing_text_path': existing_text_path,
            'status_label': status_label,
            'preview_output': preview_output,
            'row': row,
        })

    def mark_saved(entry) -> None:
        entry['row'].layout.border_left = f'3px solid {saved_color}'
        entry['status_label'].value = (
            f'<span style="color:{saved_color}; font-size:0.85em;">&#10003; saved</span>')

    def mark_empty(entry) -> None:
        entry['row'].layout.border_left = f'3px solid {border_color}'
        entry['status_label'].value = ''

    def mark_error(entry, error: Exception) -> None:
        entry['row'].layout.border_left = '3px solid #cc0000'
        entry['status_label'].value = (
            f'<span style="color:#cc0000; font-size:0.85em;">&#10007; not saved: {error}</span>')

    def save_entry(entry) -> str:
        """Persist one box. Returns 'saved', 'blank' or 'error'."""
        content = entry['text_area'].value.strip()
        if not content:
            mark_empty(entry)
            entry['preview_output'].clear_output()
            return 'blank'
        try:
            save_typed_answer(content, entry['text_path'], entry['image_path'])
        except Exception as error:
            mark_error(entry, error)
            return 'error'
        mark_saved(entry)
        with entry['preview_output']:
            entry['preview_output'].clear_output(wait=True)
            display(Markdown(content))
        return 'saved'

    for entry in entries:
        # Rehydrate a previously saved answer, then wire up autosave-on-blur.
        if entry['existing_text_path'].exists():
            try:
                saved_text = entry['existing_text_path'].read_text(encoding='utf-8')
                if saved_text.startswith('---'):
                    saved_text = saved_text.split('---', 2)[2].strip()
                entry['text_area'].value = saved_text
                mark_saved(entry)
                with entry['preview_output']:
                    display(Markdown(saved_text))
            except Exception as error:
                mark_error(entry, error)

        def on_blur(change, entry=entry):
            save_entry(entry)

        entry['text_area'].observe(on_blur, names='value')

    save_all_button = widgets.Button(
        description='Save all answers',
        button_style='success',
        icon='check',
        layout=widgets.Layout(width='180px'),
    )
    status_output = widgets.Output()

    def save_all(button):
        counts = {'saved': 0, 'blank': 0, 'error': 0}
        for entry in entries:
            counts[save_entry(entry)] += 1
        with status_output:
            status_output.clear_output(wait=True)
            message = f"\u2713 Saved {counts['saved']} answer(s)."
            if counts['blank']:
                message += f"  {counts['blank']} left blank."
            if counts['error']:
                message += f"  {counts['error']} error(s) -- see above."
            print(message)

    save_all_button.on_click(save_all)

    total_points = sum(p['points'] for p in prompts if p.get('points') is not None)
    title_bits = [f'<b>Short answers &mdash; {block_key}</b>']
    if total_points:
        title_bits.append(f'{len(prompts)} prompts, {total_points} points total')
    else:
        title_bits.append(f'{len(prompts)} prompts')
    title = widgets.HTML(
        value=(f'<div style="border-bottom:1px solid {border_color}; padding-bottom:6px; '
               f'margin-bottom:4px;">{title_bits[0]}'
               f'<span style="color:{muted_color}; font-size:0.85em; padding-left:10px;">'
               f'{title_bits[1]} &middot; answers save when you click away</span></div>')
    )

    display(widgets.VBox(
        [title] + rows + [save_all_button, status_output],
        layout=widgets.Layout(padding='10px 10px 10px 4px', margin='5px 0'),
    ))


def _escape_markdown_table_cell(value: str) -> str:
    """Keep arbitrary student text inside one generated Markdown table cell."""
    return str(value).strip().replace('|', r'\|').replace('\n', '<br>')


def _render_table_answer_markdown(schema: dict, responses: dict) -> str:
    """Render a table-answer response as Markdown for the submission PDF."""
    columns = schema.get('columns', [])
    rows = schema.get('rows', [])
    headings = ['Control'] + [column.get('label', column.get('id', '')) for column in columns]
    output = [
        '| ' + ' | '.join(_escape_markdown_table_cell(value) for value in headings) + ' |',
        '| ' + ' | '.join(['---'] * len(headings)) + ' |',
    ]
    for row in rows:
        row_values = responses.get(row.get('id', ''), {})
        values = [row.get('label', row.get('id', ''))]
        values.extend(row_values.get(column.get('id', ''), '') for column in columns)
        output.append(
            '| ' + ' | '.join(_escape_markdown_table_cell(value) for value in values) + ' |')
    return '\n'.join(output)


def _write_table_answer_files(answer_id: str, schema: dict, responses: dict) -> int:
    """Persist structured table data plus its derived Markdown; return the blank-cell count."""
    import json

    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    normalized = {}
    blank_count = 0
    for row in schema.get('rows', []):
        row_id = row.get('id', '')
        normalized[row_id] = {}
        source_row = responses.get(row_id, {})
        for column in schema.get('columns', []):
            column_id = column.get('id', '')
            value = str(source_row.get(column_id, '')).strip()
            normalized[row_id][column_id] = value
            blank_count += int(not value)

    timestamp = datetime.now().isoformat()
    json_path = ANSWERS_DIR / f"{answer_id}.json"
    markdown_path = ANSWERS_DIR / f"{answer_id}.md"
    json_path.write_text(json.dumps({
        'schema_version': 1,
        'answer_id': answer_id,
        'updated': timestamp,
        'responses': normalized,
    }, indent=1, ensure_ascii=False), encoding='utf-8')
    markdown_path.write_text(
        '---\n'
        'answer_type: table\n'
        f'created: {timestamp}\n'
        '---\n'
        f'{_render_table_answer_markdown(schema, normalized)}\n',
        encoding='utf-8',
    )
    stale_image = ANSWERS_DIR / f"{answer_id}.png"
    if stale_image.exists():
        stale_image.unlink()
    return blank_count


def add_table_answer(answer_id: str) -> None:
    """Render and autosave a schema-driven ``@@TABLE_ANSWER`` response table.

    Schemas are generated at build time in ``Files/table_answer_schemas.json``. Structured values
    are saved to ``Answers/{answer_id}.json`` for lossless reload and validation; a derived
    ``.md`` table is saved beside it for the existing submission-PDF renderer.
    """
    import json

    schemas_path = Path(__file__).parent / 'table_answer_schemas.json'
    try:
        schemas = json.loads(schemas_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        display(widgets.HTML(
            f'<b style="color:#cc0000;">table_answer_schemas.json not found at '
            f'{schemas_path}</b>'))
        return
    except (OSError, json.JSONDecodeError) as error:
        display(widgets.HTML(
            f'<b style="color:#cc0000;">Could not read table-answer schemas: {error}</b>'))
        return

    schema = schemas.get(str(answer_id))
    if not isinstance(schema, dict):
        display(widgets.HTML(
            f'<b style="color:#cc0000;">No table-answer schema found for "{answer_id}".</b>'))
        return
    columns = schema.get('columns', [])
    rows = schema.get('rows', [])
    if not columns or not rows:
        display(widgets.HTML(
            f'<b style="color:#cc0000;">The table-answer schema for "{answer_id}" is empty.</b>'))
        return

    initial_responses = {}
    existing_json = _existing_answer_path(answer_id, '.json')
    if existing_json.exists():
        try:
            payload = json.loads(existing_json.read_text(encoding='utf-8'))
            if isinstance(payload.get('responses'), dict):
                initial_responses = payload['responses']
        except (OSError, json.JSONDecodeError):
            initial_responses = {}

    border_color = 'var(--jp-border-color1, #dddddd)'
    muted_color = 'var(--jp-ui-font-color2, #666666)'
    field_widgets = {}

    def markdown_box(source: str, *, width: str, bold: bool = False):
        output = widgets.Output(layout=widgets.Layout(width=width, padding='4px 6px'))
        with output:
            display(Markdown(f'**{source}**' if bold else source))
        return output

    label_width = '18%'
    field_width = f'{82 / len(columns):.3g}%'
    header = [widgets.HTML(value='<b>Control</b>',
                           layout=widgets.Layout(width=label_width, padding='4px 6px'))]
    header.extend(markdown_box(column.get('label', column.get('id', '')),
                               width=field_width, bold=True)
                  for column in columns)
    table_rows = [widgets.HBox(
        header,
        layout=widgets.Layout(width='100%', border_bottom=f'1px solid {border_color}'),
    )]

    for row in rows:
        row_id = row.get('id', '')
        row_widgets = [markdown_box(row.get('label', row_id), width=label_width, bold=True)]
        stored_row = initial_responses.get(row_id, {})
        for column in columns:
            column_id = column.get('id', '')
            initial_value = str(stored_row.get(column_id, '')).strip()
            if column.get('type') == 'choice':
                options = [('', '')] + [(str(option), str(option))
                                        for option in column.get('options', [])]
                valid_values = {value for _label, value in options}
                if initial_value not in valid_values:
                    initial_value = ''
                field = widgets.Dropdown(
                    options=options,
                    value=initial_value,
                    description='',
                    layout=widgets.Layout(width=field_width, padding='2px 6px'),
                )
            else:
                field = widgets.Textarea(
                    value=initial_value,
                    rows=max(1, int(column.get('lines', 2))),
                    continuous_update=False,
                    layout=widgets.Layout(width=field_width, padding='2px 6px'),
                )
            field_widgets[(row_id, column_id)] = field
            row_widgets.append(field)
        table_rows.append(widgets.HBox(
            row_widgets,
            layout=widgets.Layout(width='100%', align_items='stretch',
                                  border_bottom=f'1px solid {border_color}', padding='4px 0'),
        ))

    status = widgets.HTML(value='')

    def current_responses() -> dict:
        return {
            row.get('id', ''): {
                column.get('id', ''): field_widgets[
                    (row.get('id', ''), column.get('id', ''))].value
                for column in columns
            }
            for row in rows
        }

    def save_table(_change=None) -> None:
        try:
            blanks = _write_table_answer_files(answer_id, schema, current_responses())
        except Exception as error:
            status.value = (
                f'<span style="color:#cc0000;">&#10007; table not saved: {error}</span>')
            return
        suffix = f' &middot; {blanks} cell(s) still blank' if blanks else ''
        status.value = (
            f'<span style="color:#28a745;">&#10003; table saved</span>'
            f'<span style="color:{muted_color};">{suffix}</span>')

    for field in field_widgets.values():
        field.observe(save_table, names='value')

    save_button = widgets.Button(
        description='Save table', button_style='success', icon='check',
        layout=widgets.Layout(width='150px'),
    )
    save_button.on_click(save_table)
    title = widgets.HTML(
        value=(f'<b>Table answer &mdash; {answer_id}</b>'
               f'<span style="color:{muted_color}; font-size:0.85em; padding-left:10px;">'
               'changes save when you leave a field</span>'))
    display(widgets.VBox(
        [title] + table_rows + [widgets.HBox([save_button, status])],
        layout=widgets.Layout(border=f'1px solid {border_color}', padding='10px', margin='5px 0'),
    ))
