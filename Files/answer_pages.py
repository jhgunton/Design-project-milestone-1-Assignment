"""Read ordered answer-selector pages in current and legacy formats.

This small module is also copied into a released notebook's Files directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


PAGE_MANIFEST_SUFFIX = ".pages.json"
IMAGE_MANIFEST_SUFFIX = ".images.json"


@dataclass(frozen=True)
class AnswerPage:
    kind: str
    path: Path


@dataclass(frozen=True)
class AnswerResponse:
    pages: tuple[AnswerPage, ...] = ()

    @property
    def image_paths(self) -> tuple[Path, ...]:
        """Compatibility view for callers that inspect image-only answers."""
        return tuple(page.path for page in self.pages if page.kind == "image")

    @property
    def text_path(self) -> Path | None:
        """Compatibility view for a single typed answer."""
        texts = [page.path for page in self.pages if page.kind == "text"]
        return texts[0] if len(self.pages) == 1 and len(texts) == 1 else None


def read_answer_response(root: Path, answer_id: str) -> AnswerResponse:
    """Prefer Answers/ over Images/ and preserve each format's saved order."""
    for directory in (root / "Answers", root / "Images"):
        page_manifest = directory / f"{answer_id}{PAGE_MANIFEST_SUFFIX}"
        image_manifest = directory / f"{answer_id}{IMAGE_MANIFEST_SUFFIX}"
        text_path = directory / f"{answer_id}.md"
        image_path = directory / f"{answer_id}.png"
        if page_manifest.exists():
            try:
                payload = json.loads(page_manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read answer-page manifest {page_manifest}: {exc}") from exc
            entries = payload.get("pages") if isinstance(payload, dict) else None
            if not isinstance(payload, dict) or payload.get("schema_version") != 2 or not isinstance(entries, list):
                raise ValueError(f"Invalid answer-page manifest: {page_manifest}")
            pages = []
            seen = set()
            for entry in entries:
                kind = entry.get("type") if isinstance(entry, dict) else None
                filename = entry.get("file") if isinstance(entry, dict) else None
                suffix = ".png" if kind == "image" else ".md" if kind == "text" else None
                if (suffix is None or not isinstance(filename, str) or
                        Path(filename).name != filename or
                        not filename.startswith(f"{answer_id}_page_") or
                        not filename.endswith(suffix) or filename in seen):
                    raise ValueError(f"Unsafe or unsupported answer page {entry!r} in {page_manifest}")
                path = directory / filename
                if not path.is_file():
                    raise ValueError(f"Answer-page manifest refers to missing file: {path}")
                seen.add(filename)
                pages.append(AnswerPage(kind, path))
            return AnswerResponse(tuple(pages))
        if text_path.exists():
            return AnswerResponse((AnswerPage("text", text_path),))
        if image_manifest.exists():
            try:
                payload = json.loads(image_manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read image-page manifest {image_manifest}: {exc}") from exc
            names = payload.get("pages") if isinstance(payload, dict) else None
            if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(names, list) or not names:
                raise ValueError(f"Invalid image-page manifest: {image_manifest}")
            pages = []
            for name in names:
                if (not isinstance(name, str) or Path(name).name != name or
                        not name.lower().endswith(".png")):
                    raise ValueError(f"Unsafe or unsupported page filename {name!r} in {image_manifest}")
                path = directory / name
                if not path.is_file():
                    raise ValueError(f"Image-page manifest refers to missing file: {path}")
                pages.append(AnswerPage("image", path))
            return AnswerResponse(tuple(pages))
        if image_path.exists():
            return AnswerResponse((AnswerPage("image", image_path),))
    return AnswerResponse()
