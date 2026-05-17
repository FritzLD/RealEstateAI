"""
extract_reference_docs.py
=========================
Run this script locally once after adding or updating PDFs in docs/.
It chunks the text and saves to data/reference_chunks.json, which the
Streamlit Cloud app loads instantly — no PDF parsing at runtime.

Usage:
    cd "D:\\Real estate Forecast\\RealEstateAI"
    pip install pypdf langchain-text-splitters   # first time only
    python scripts/extract_reference_docs.py

Output:
    data/reference_chunks.json   <- commit this file and push to GitHub
"""
# this docstream explains the script and how to run it 

from __future__ import annotations
# allowing future scripts to work more smoothly 
import json # save extracted chunks as a JSON file
import sys # used to modify python import path
import warnings # suppress warning messages
from datetime import datetime # timestamp when the extraction runs
from pathlib import Path # makes file and folder paths easier to work with 

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# adds the project root folder to Python's import path.

warnings.filterwarnings("ignore")
# hides messages

from langchain_community.document_loaders import PyPDFLoader
# loads pdf files into Langchains document objects
from langchain_text_splitters import RecursiveCharacterTextSplitter
# splits long document text into smaller chunks

from src import config

# ── Constants ─────────────────────────────────────────────────────────────────

DOCS_DIR    = config.PROJECT_ROOT / "docs" # merely points to where the documents are located
OUTPUT_FILE = config.DATA_DIR / "reference_chunks.json" # sets where the extracted chunks will be saved

CHUNK_SIZE    = 1000 # sets the target size of each text chunk
CHUNK_OVERLAP = 150 # keeps some overlap between chunks to preserve context


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None: # defines the main script function
    print("=" * 60)
    print("RealEstateAI – Reference Document Extractor")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    if not DOCS_DIR.exists(): # just checking to see if the pdf files are in the folder
        print(f"  docs/ directory not found at {DOCS_DIR}")
        return

    pdfs = sorted(DOCS_DIR.glob("*.pdf")) # finds all the pdf files in the docs folder
    if not pdfs: # if pdfs are found or not
        print("  No PDF files found in docs/")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    # creates a text splitter

    all_chunks: list[dict] = [] # create a empty list to store all extracted chunks

    for pdf_path in pdfs:
        print(f"\n  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            all_chunks.append({
                "page_content": chunk.page_content,
                "metadata": {
                    **chunk.metadata,
                    "source_file": pdf_path.name,
                    "type": "reference_document",
                },
            })

        print(f"  -> {len(pages)} pages -> {len(chunks)} chunks")

    output = {
        "extracted_at":   datetime.now().isoformat(),
        "chunk_size":     CHUNK_SIZE,
        "chunk_overlap":  CHUNK_OVERLAP,
        "total_chunks":   len(all_chunks),
        "chunks":         all_chunks,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_chars = sum(len(c["page_content"]) for c in all_chunks)
    print(f"\n  Extracted {len(all_chunks):,} chunks ({total_chars:,} characters)")
    print(f"  Saved -> {OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  git add data/reference_chunks.json")
    print('  git commit -m "Add pre-extracted reference document chunks"')
    print("  git push")
    print("=" * 60)


if __name__ == "__main__":
    main()

# overall this takes the pdf guidelines and breaks them into searchable chunks, and saves them into reference_chunks.json so that the streamlit app
# does not have to parse the PDF's everytime it runs
