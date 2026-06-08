#!/usr/bin/env python3
"""Strip all metadata from the resume PDF and set a clean title.

Usage:
    python3 scripts/clean_resume_pdf.py            # cleans files/Ishaan_Vohra_resume.pdf in place
    python3 scripts/clean_resume_pdf.py path.pdf   # cleans the given file in place

Removes the document Info dictionary and the XMP metadata stream, then sets
the document title to "Ishaan Vohra - Resume".
"""
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, create_string_object

TITLE = "Ishaan Vohra - Resume"
DEFAULT = Path(__file__).resolve().parent.parent / "files" / "Ishaan_Vohra_resume.pdf"


def clean(path: Path) -> None:
    reader = PdfReader(str(path))
    writer = PdfWriter()
    writer.append(reader)

    # Drop any inherited Info dict entries, then set only the Title.
    writer.add_metadata({"/Title": TITLE})

    # Remove the XMP metadata stream from the catalog (where viewers often
    # read the title/author from instead of the Info dict).
    root = writer._root_object
    if "/Metadata" in root:
        del root[NameObject("/Metadata")]

    # Ensure the Info dict contains exactly Title (and Producer pypdf adds).
    info = writer._info.get_object()
    for key in list(info.keys()):
        if key != "/Title":
            del info[key]
    info[NameObject("/Title")] = create_string_object(TITLE)

    with open(path, "wb") as f:
        writer.write(f)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"error: {path} not found. Upload the PDF there first.", file=sys.stderr)
        return 1
    clean(path)
    print(f"Cleaned metadata and set title to '{TITLE}' on {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
