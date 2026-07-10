"""
chunker.py
----------
Converts raw 10-K HTML filings and earnings call transcript HTML pages
into clean, structured chunks with metadata.

Two chunking strategies:
  1. TenKChunker   — for SEC EDGAR 10-K HTML files
  2. TranscriptChunker — for Motley Fool earnings call transcript HTML pages

Each chunk is a dict:
{
    "chunk_id":   "jpm_10k_2025_007",
    "text":       "...",
    "metadata": {
        "company":    "JPM",
        "doc_type":   "10-K" | "earnings_transcript",
        "year":       "2025",
        "section":    "Item 7",          # 10-K only
        "chunk_type": "narrative"|"table", # 10-K only
        "turn":       3,                  # transcript only
        "speakers":   "Jeremy Barnum",    # transcript only
        "source":     "jpm10K-20251231.html"
    }
}
"""

import re
import json
import warnings
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ── constants ────────────────────────────────────────────────────────────────
CHUNK_SIZE_WORDS   = 400   # target words per narrative chunk
CHUNK_OVERLAP_WORDS = 60   # overlap between consecutive narrative chunks
TABLE_MAX_WORDS    = 600   # tables larger than this get split

# ── helpers ──────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _sliding_window(text: str, size: int = CHUNK_SIZE_WORDS,
                    overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


def _clean_whitespace(text: str) -> str:
    """Collapse multiple blank lines / spaces."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _strip_hidden_and_xbrl(soup: BeautifulSoup) -> None:
    """Remove XBRL metadata blocks and display:none elements in-place."""
    for tag in soup.find_all('ix:header'):
        tag.decompose()
    for tag in soup.find_all(
        style=lambda v: v and 'display:none' in v.replace(' ', '').lower()
    ):
        tag.decompose()


# ── 10-K chunker ─────────────────────────────────────────────────────────────

# SEC 10-K sections we actually care about for a finance RAG
RELEVANT_ITEMS = {
    "Item 1":  "Business Overview",
    "Item 1A": "Risk Factors",
    "Item 7":  "Management Discussion and Analysis",
    "Item 7A": "Market Risk",
    "Item 8":  "Financial Statements",
}

ITEM_PATTERN = re.compile(
    r'Item\s+(1A|1B|1C|1|7A|7|8|9A|9|2|3|4|5|6|10|11|12|13|14|15)\.?',
    re.IGNORECASE
)


def _table_to_text(table_tag) -> str:
    """Convert an HTML <table> into a readable pipe-delimited text block."""
    rows = []
    for tr in table_tag.find_all('tr'):
        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
        if any(cells):  # skip fully empty rows
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def chunk_10k(html_path: Path, company: str, year: str) -> list[dict]:
    """
    Parse a 10-K HTML file and return a list of chunks.
    - Tables → preserved as 'table' type chunks
    - Narrative text → sliding-window 'narrative' chunks
    Both are tagged with company, year, and Item section.
    """
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'lxml')

    _strip_hidden_and_xbrl(soup)

    source = html_path.name
    chunks = []
    chunk_index = 0

    def make_id():
        nonlocal chunk_index
        cid = f"{company.lower()}_10k_{year}_{chunk_index:04d}"
        chunk_index += 1
        return cid

    # ── Step 1: pull out all tables, replace with placeholder ────────────────
    table_chunks = []
    tables = soup.find_all('table')
    for i, table in enumerate(tables):
        table_text = _table_to_text(table)
        if _word_count(table_text) < 10:        # skip trivial/layout tables
            table.decompose()
            continue

        # Try to grab the nearest preceding heading/bold text as table title
        prev = table.find_previous(['h1','h2','h3','h4','b','strong'])
        title = prev.get_text(strip=True)[:80] if prev else f"Financial Table {i+1}"

        # Replace table in DOM with a placeholder so narrative flow stays intact
        placeholder = soup.new_tag('p')
        placeholder.string = f"[TABLE:{i}]"
        table.replace_with(placeholder)

        table_chunks.append((f"[TABLE:{i}]", title, table_text))

    # ── Step 2: extract clean narrative text ─────────────────────────────────
    full_text = _clean_whitespace(soup.get_text(separator='\n'))

    # Skip table-of-contents (first occurrence of "Item 1." with page numbers)
    # Actual content starts at the second occurrence
    item_positions = [(m.start(), m.group()) for m in ITEM_PATTERN.finditer(full_text)]

    # Find where the TOC ends and real content begins
    # TOC entries are closely packed; real sections are far apart
    toc_end = 0
    for j in range(1, len(item_positions)):
        if item_positions[j][0] - item_positions[j-1][0] > 5000:
            toc_end = item_positions[j][0]
            break

    content_text = full_text[toc_end:] if toc_end else full_text

    # ── Step 3: split narrative into Item sections ────────────────────────────
    section_splits = list(ITEM_PATTERN.finditer(content_text))
    sections = []
    for k, match in enumerate(section_splits):
        start = match.start()
        end = section_splits[k+1].start() if k+1 < len(section_splits) else len(content_text)
        label = match.group().strip().rstrip('.')
        sections.append((label, content_text[start:end]))

    if not sections:
        sections = [("Full Document", content_text)]

    # ── Step 4: chunk each section ────────────────────────────────────────────
    for section_label, section_text in sections:
        section_name = RELEVANT_ITEMS.get(section_label, section_label)

        # Resolve table placeholders within this section
        def embed_tables(text):
            """Replace [TABLE:N] placeholders with actual table text."""
            result = text
            for placeholder, title, table_text in table_chunks:
                if placeholder in result:
                    replacement = f"\n[Financial Table — {title}]\n{table_text}\n"
                    result = result.replace(placeholder, replacement, 1)
            return result

        section_text = embed_tables(section_text)

        # Split section into paragraphs first, then apply sliding window
        paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
        buffer = []
        buffer_words = 0

        for para in paragraphs:
            para_words = _word_count(para)

            # If a single paragraph is huge (e.g. a large table), split it
            if para_words > CHUNK_SIZE_WORDS * 2:
                if buffer:
                    # flush buffer first
                    for window in _sliding_window(" ".join(buffer)):
                        chunks.append({
                            "chunk_id": make_id(),
                            "text": window,
                            "metadata": {
                                "company":    company,
                                "doc_type":   "10-K",
                                "year":       year,
                                "section":    section_label,
                                "section_name": section_name,
                                "chunk_type": "narrative",
                                "source":     source,
                            }
                        })
                    buffer = []
                    buffer_words = 0
                # chunk the big paragraph directly
                for window in _sliding_window(para):
                    chunk_type = "table" if "[Financial Table" in window else "narrative"
                    chunks.append({
                        "chunk_id": make_id(),
                        "text": window,
                        "metadata": {
                            "company":    company,
                            "doc_type":   "10-K",
                            "year":       year,
                            "section":    section_label,
                            "section_name": section_name,
                            "chunk_type": chunk_type,
                            "source":     source,
                        }
                    })
            else:
                buffer.append(para)
                buffer_words += para_words
                if buffer_words >= CHUNK_SIZE_WORDS:
                    chunk_text = " ".join(buffer)
                    chunks.append({
                        "chunk_id": make_id(),
                        "text": chunk_text,
                        "metadata": {
                            "company":    company,
                            "doc_type":   "10-K",
                            "year":       year,
                            "section":    section_label,
                            "section_name": section_name,
                            "chunk_type": "narrative",
                            "source":     source,
                        }
                    })
                    # keep last paragraph as overlap seed
                    buffer = [buffer[-1]]
                    buffer_words = _word_count(buffer[0])

        # flush remaining buffer
        if buffer:
            chunks.append({
                "chunk_id": make_id(),
                "text": " ".join(buffer),
                "metadata": {
                    "company":    company,
                    "doc_type":   "10-K",
                    "year":       year,
                    "section":    section_label,
                    "section_name": section_name,
                    "chunk_type": "narrative",
                    "source":     source,
                }
            })

    return chunks


# ── Transcript chunker ────────────────────────────────────────────────────────

# Known executive speakers (used to tag who is answering)
EXEC_SPEAKERS = re.compile(
    r'(Jamie Dimon|James Dimon|Jeremy Barnum|Mary Erdoes|Daniel Pinto|'
    r'Tim Cook|Luca Maestri|Kevan Parekh|'         # Apple
    r'Bob Jordan|Tammy Romo|'                        # Southwest
    r'Operator)',
    re.IGNORECASE
)

SPEAKER_LINE = re.compile(r'^([A-Z][A-Za-z\s\.\-]+):\s*$', re.MULTILINE)


def chunk_transcript(html_path: Path, company: str, quarter: str) -> list[dict]:
    """
    Parse a Motley Fool earnings call transcript HTML and return chunks.
    Each chunk = one speaker turn (or a Q&A pair grouped together).
    """
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'lxml')

    for tag in soup.find_all(
        style=lambda v: v and 'display:none' in v.replace(' ', '').lower()
    ):
        tag.decompose()

    text = soup.get_text(separator='\n', strip=True)
    source = html_path.name

    # ── Find transcript content boundaries ───────────────────────────────────
    # Start: first mention of a known participant or "Prepared Remarks"
    start_markers = ['Prepared Remarks', 'Call Participants', 'Operator:',
                     'Jeremy Barnum', 'Jamie Dimon', 'Tim Cook', 'Bob Jordan']
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = idx
            break

    # End: Motley Fool footer
    end_markers = ['Making the world smarter', 'Premium Investing Services',
                   '© 1995', 'Stocks Mentioned']
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    transcript_text = text[start_idx:end_idx]

    # ── Parse speaker turns ──────────────────────────────────────────────────
    # Motley Fool format: speaker name on its own line, then dialogue below
    lines = transcript_text.split('\n')
    turns = []   # list of (speaker, text)
    current_speaker = "Unknown"
    current_lines = []

    # Known speaker name patterns (standalone lines)
    name_pattern = re.compile(
        r'^(Jeremy Barnum|Jamie Dimon|James Dimon|Operator|'
        r'Mary Caldwell Erdoes|Daniel Pinto|Tim Cook|Luca Maestri|Kevan Parekh|'
        r'Bob Jordan|Tammy Romo|'
        r'[A-Z][a-z]+ [A-Z][a-z]+)$'   # generic "First Last" pattern
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect speaker lines — either "Name:" format or standalone "First Last"
        if line.endswith(':') and len(line) < 50:
            speaker_name = line.rstrip(':')
            if current_lines:
                turns.append((current_speaker, _clean_whitespace('\n'.join(current_lines))))
            current_speaker = speaker_name
            current_lines = []
        elif name_pattern.match(line) and len(line) < 50:
            if current_lines:
                turns.append((current_speaker, _clean_whitespace('\n'.join(current_lines))))
            current_speaker = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        turns.append((current_speaker, _clean_whitespace('\n'.join(current_lines))))

    # ── Group into Q&A pairs ─────────────────────────────────────────────────
    # Strategy: Operator intro + Analyst question + all executive replies = 1 chunk
    chunks = []
    chunk_index = 0

    def make_id():
        nonlocal chunk_index
        cid = f"{company.lower()}_transcript_{quarter}_{chunk_index:04d}"
        chunk_index += 1
        return cid

    i = 0
    while i < len(turns):
        speaker, content = turns[i]

        if speaker.lower() == 'operator' or 'operator' in speaker.lower():
            # Start of a Q&A exchange — gather intro + analyst + exec responses
            qa_block = [f"Operator: {content}"]
            speakers_in_block = ["Operator"]
            i += 1

            while i < len(turns):
                spk, txt = turns[i]
                is_exec = bool(EXEC_SPEAKERS.match(spk)) or 'operator' in spk.lower()
                qa_block.append(f"{spk}: {txt}")
                speakers_in_block.append(spk)
                i += 1
                # Stop after executive response(s), before next Operator turn
                if is_exec and i < len(turns) and 'operator' in turns[i][0].lower():
                    break

            chunks.append({
                "chunk_id": make_id(),
                "text": "\n\n".join(qa_block),
                "metadata": {
                    "company":    company,
                    "doc_type":   "earnings_transcript",
                    "quarter":    quarter,
                    "turn":       chunk_index - 1,
                    "speakers":   ", ".join(dict.fromkeys(speakers_in_block)),
                    "chunk_type": "qa_exchange",
                    "source":     source,
                }
            })
        else:
            # Prepared remarks or standalone speaker turn
            if _word_count(content) > CHUNK_SIZE_WORDS:
                # Long prepared remarks get windowed
                for window in _sliding_window(content):
                    chunks.append({
                        "chunk_id": make_id(),
                        "text": f"{speaker}: {window}",
                        "metadata": {
                            "company":    company,
                            "doc_type":   "earnings_transcript",
                            "quarter":    quarter,
                            "turn":       chunk_index - 1,
                            "speakers":   speaker,
                            "chunk_type": "prepared_remarks",
                            "source":     source,
                        }
                    })
            else:
                chunks.append({
                    "chunk_id": make_id(),
                    "text": f"{speaker}: {content}",
                    "metadata": {
                        "company":    company,
                        "doc_type":   "earnings_transcript",
                        "quarter":    quarter,
                        "turn":       chunk_index - 1,
                        "speakers":   speaker,
                        "chunk_type": "prepared_remarks",
                        "source":     source,
                    }
                })
            i += 1

    return chunks


# ── Runner ────────────────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(chunks)} chunks → {output_path}")


if __name__ == "__main__":
    RAW = Path("data/raw")
    OUT = Path("data/chunks")
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Known documents: (filename, company, type, year/quarter) ─────────────
    # Add or remove entries here as your corpus changes.
    # If a file is missing from data/raw/, it is skipped automatically.
    CORPUS = [
        # JPM
        ("jpm10K-20251231.html",                "JPM",  "10k",        "2025"),
        ("jpm_10k_2025.html",                   "JPM",  "10k",        "2025"),  # alt name
        ("JPM-2026_Earnings_Call_Transcript1.html","JPM","transcript", "Q1_2026"),
        # AAPL
        ("aapl_10k_2024.html",                  "AAPL", "10k",        "2024"),
        ("aapl_10k_2025.html",                  "AAPL", "10k",        "2025"),
        # LUV
        ("luv_10k_2025.html",                   "LUV",  "10k",        "2025"),
        ("luv_10k_2026.html",                   "LUV",  "10k",        "2026"),
    ]

    all_chunks = []
    seen_outputs = set()   # avoid double-chunking if two filenames map to same output

    for filename, company, doc_type, period in CORPUS:
        path = RAW / filename
        if not path.exists():
            print(f"  SKIP (not found): {filename}")
            continue

        if doc_type == "10k":
            out_name = f"{company.lower()}_10k_{period}.json"
        else:
            out_name = f"{company.lower()}_transcript_{period}.json"

        if out_name in seen_outputs:
            print(f"  SKIP (already processed): {filename}")
            continue
        seen_outputs.add(out_name)

        print(f"\nChunking {filename} ...")
        try:
            if doc_type == "10k":
                chunks = chunk_10k(html_path=path, company=company, year=period)
            else:
                chunks = chunk_transcript(html_path=path, company=company, quarter=period)

            save_chunks(chunks, OUT / out_name)
            all_chunks.extend(chunks)

        except Exception as e:
            print(f"  ERROR chunking {filename}: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"DONE — {len(all_chunks)} total chunks")
    print(f"{'='*50}")
    for name in sorted(seen_outputs):
        path = OUT / name
        if path.exists():
            with open(path) as f:
                n = len(json.load(f))
            print(f"  {name:<45} {n:>5} chunks")
