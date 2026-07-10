"""
download_corpus.py  (v2 — fixed document detection)
-----------------------------------------------------
Downloads the two most recent 10-K filings for JPM, AAPL, LUV
directly from SEC EDGAR. Picks the main filing document by
selecting the LARGEST .htm file in the filing index (which is
always the actual 10-K, not a small exhibit or wrapper).

Run from project root:
    pip install requests
    python scripts/download_corpus.py
"""

import time
import requests
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
COMPANIES = {
    "JPM":  {"name": "JPMorgan Chase",     "cik": "0000019617"},
    "AAPL": {"name": "Apple Inc.",         "cik": "0000320193"},
    "LUV":  {"name": "Southwest Airlines", "cik": "0000092380"},
}

NUM_FILINGS = 2
OUTPUT_DIR  = Path("data/raw")

# !! IMPORTANT: put your real name + email here (SEC policy, not auth)
HEADERS_DATA = {
    "User-Agent":       "Arshia Garg arshiagarg@example.com",
    "Accept-Encoding":  "gzip, deflate",
    "Host":             "data.sec.gov",
}
HEADERS_WWW = {
    "User-Agent":       "Arshia Garg arshiagarg@example.com",
    "Accept-Encoding":  "gzip, deflate",
    "Host":             "www.sec.gov",
}

DATA_BASE = "https://data.sec.gov"
SEC_BASE  = "https://www.sec.gov"

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_recent_10k_accessions(cik: str, n: int = 2) -> list[dict]:
    """Get n most recent 10-K accession numbers from EDGAR submissions API."""
    url = f"{DATA_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS_DATA, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent   = data["filings"]["recent"]
    forms    = recent["form"]
    accnos   = recent["accessionNumber"]
    dates    = recent["filingDate"]
    pri_docs = recent["primaryDocument"]

    results = []
    for i, form in enumerate(forms):
        if form == "10-K":
            results.append({
                "accession":   accnos[i],
                "date":        dates[i],
                "primary_doc": pri_docs[i],
            })
            if len(results) == n:
                break
    return results


def get_all_filing_files(cik: str, accession: str) -> list[dict]:
    """
    Fetch the filing index JSON and return all files with their sizes.
    EDGAR index.json structure:
      { "directory": { "item": [ {"name":..., "type":..., "size":...}, ... ] } }
    """
    acc_nodash = accession.replace("-", "")
    cik_num    = cik.lstrip("0")
    url = f"{SEC_BASE}/Archives/edgar/data/{cik_num}/{acc_nodash}/index.json"
    resp = requests.get(url, headers=HEADERS_WWW, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("directory", {}).get("item", [])


def pick_main_10k_htm(files: list[dict], primary_doc: str) -> str:
    """
    Strategy: the main 10-K document is always the LARGEST .htm file.
    Exhibit wrappers and viewer files are all tiny (< 50 KB).
    The actual iXBRL 10-K is always several MB.
    """
    # Filter to .htm files only
    htm_files = [
        f for f in files
        if f["name"].lower().endswith(".htm")
        and not f["name"].lower().startswith("r")   # skip R1.htm, R2.htm report files
        and "exhibit" not in f["name"].lower()
        and "ex" not in f["name"].lower()[:4]
    ]

    if not htm_files:
        return primary_doc  # last resort fallback

    # Sort by size descending — main doc is always largest
    def parse_size(f):
        try:
            return int(f.get("size", 0))
        except (ValueError, TypeError):
            return 0

    htm_files.sort(key=parse_size, reverse=True)

    print(f"    Top .htm files by size:")
    for f in htm_files[:5]:
        size_kb = parse_size(f) / 1024
        print(f"      {f['name']:<45} {size_kb:>8.1f} KB  type={f.get('type','?')}")

    largest = htm_files[0]["name"]
    print(f"    → Selected: {largest}")
    return largest


def download_file(cik: str, accession: str, filename: str, out_path: Path) -> int:
    """Download a filing document. Returns bytes downloaded."""
    acc_nodash = accession.replace("-", "")
    cik_num    = cik.lstrip("0")
    url = f"{SEC_BASE}/Archives/edgar/data/{cik_num}/{acc_nodash}/{filename}"

    print(f"    URL: {url}")
    resp = requests.get(url, headers=HEADERS_WWW, stream=True, timeout=60)

    if resp.status_code != 200:
        print(f"    ERROR: HTTP {resp.status_code}")
        return 0

    content_type = resp.headers.get("Content-Type", "")
    print(f"    Content-Type: {content_type}")

    total = 0
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=131072):  # 128KB chunks
            f.write(chunk)
            total += len(chunk)
            print(f"\r    {total / 1_000_000:.2f} MB downloaded...", end="", flush=True)

    print(f"\r    ✓ {total / 1_000_000:.2f} MB  →  {out_path.name}")
    return total


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker, info in COMPANIES.items():
        cik  = info["cik"]
        name = info["name"]
        print(f"\n{'='*60}")
        print(f"  {ticker}  —  {name}")
        print(f"{'='*60}")

        try:
            filings = get_recent_10k_accessions(cik, NUM_FILINGS)
        except Exception as e:
            print(f"  FAILED to get filings list: {e}")
            continue

        for filing in filings:
            acc  = filing["accession"]
            date = filing["date"]
            year = date[:4]
            print(f"\n  Filing: {date}  |  Accession: {acc}")

            out_name = f"{ticker.lower()}_10k_{year}.html"
            out_path = OUTPUT_DIR / out_name

            if out_path.exists() and out_path.stat().st_size > 100_000:
                print(f"  Already downloaded ({out_path.stat().st_size/1e6:.1f} MB), skipping.")
                continue

            # Get full file list from filing index
            try:
                all_files = get_all_filing_files(cik, acc)
                main_doc  = pick_main_10k_htm(all_files, filing["primary_doc"])
            except Exception as e:
                print(f"  WARNING: index fetch failed ({e}), using primary_doc")
                main_doc = filing["primary_doc"]

            # Download it
            try:
                bytes_dl = download_file(cik, acc, main_doc, out_path)
                if bytes_dl < 50_000:
                    print(f"  WARNING: file seems too small ({bytes_dl} bytes) — may be wrong doc")
            except Exception as e:
                print(f"  FAILED to download: {e}")

            time.sleep(0.6)   # be polite to SEC servers

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  DONE — files in data/raw/:")
    print(f"{'='*60}")
    for f in sorted(OUTPUT_DIR.glob("*.html")):
        mb = f.stat().st_size / 1_000_000
        status = "✓ GOOD" if mb > 1 else "⚠ SMALL — may be wrong file"
        print(f"  {f.name:<40} {mb:>6.1f} MB  {status}")


if __name__ == "__main__":
    main()
