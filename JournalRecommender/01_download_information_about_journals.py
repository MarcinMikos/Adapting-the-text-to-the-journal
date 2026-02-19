"""
Zadaniem skryptu jest pobranie danych zaciągniętych z artykułów dla wskazanych czasopism
"""

# Importowane biblioteki
import os
import re
import time
import html as html_lib
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
import multiprocessing as mp

# ----------------------------
# Konfiguracja
# ----------------------------
# API Crossref (główne źródło danych) - domyślny proces działania
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
# API Semantic Scholar (uzupełnianie abstraktów)
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper"
# API OpenAlex (alternatywne źródło)
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
# HTML Fallback (Pobieranie abstraktów bezpośrednio ze stron WWW (meta-tagi))
# Brak stałego URL - system odwiedza adresy wyciągnięte z Crossref

# Maksymalna ilość zaciąganych źródeł (artykułów) - uwaga, wartość jest mnożona dwukrotnie!
MAX_ROWS_CROSSREF = 1000

# Opóźnienia - w celu zwiększenia możliwości pobrania informacji o każdym artykule
SLEEP_CROSSREF_S = 0.6
SLEEP_S2_S = 0.6
SLEEP_OPENALEX_S = 0.3

# Limit żądań
REQ_TIMEOUT = (10, 20)  # sekundy (czas na nawiązanie połączenia, czas na przesłanie danych)

# Ograniczenia czasowe w celu niezawieszenia kodu na jednym źródle
PER_RECORD_BUDGET_S = 10          # ile maks czasu poświęcić na jeden rekord
HTML_FALLBACK_HARD_TIMEOUT_S = 10 # twardy limit na próbę HTML fallback dla jednego rekordu

# Ile URL-i z HTML fallback próbować maksymalnie dla jednego możliwego źródła - artykułu
HTML_MAX_URLS = 4

# Możliwość uruchomienia dodatkowych procesów przeszukiwania danych (wszystko poza CROSSREF)
USE_SEMANTIC_SCHOLAR = False
USE_OPENALEX = False
USE_HTML_FALLBACK = False


# ----------------------------
# Funkcje pomocnicze: czyszczenie / pozyskiwanie danych
# ----------------------------
def _strip_tags(text: str) -> str:
    if not text:
        return "" # warunek dla elementów bez tekstu - zabezpieczenie przed wystąpnieniem błędu
    text = re.sub(r"<[^>]+>", " ", text) # usunięcie znaczników
    text = html_lib.unescape(text) # dekodowanie HTML
    return re.sub(r"\s+", " ", text).strip() # redukcja białych znaków


def _normalize_doi(doi: str) -> str: # sprawdzenie sposobu zapisu DOI i weryfikacja jego istnienia
    if not doi:
        return ""
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "").strip()
    return doi


def _extract_title(item: dict) -> str: # wyciągnięcie tytułu artykułu z odpowiedzi API Crossref
    titles = item.get("title") or []
    return str(titles[0]).strip() if titles else ""


def _extract_doi(item: dict) -> str: # wyciągnięcie DOI z odpowiedzi API Crossref
    return _normalize_doi(item.get("DOI") or "")


def _extract_publication_date(item: dict) -> str: # ekstrakcja daty publikacji w formacie RRRR-MM-DD
    date_obj = item.get("published") or item.get("issued") or {}
    parts = date_obj.get("date-parts", [[]])[0]
    if not parts:
        return ""
    try:
        y = str(parts[0]).zfill(4)
        m = str(parts[1]).zfill(2) if len(parts) > 1 else "01"
        d = str(parts[2]).zfill(2) if len(parts) > 2 else "01"
        return f"{y}-{m}-{d}"
    except Exception:
        return ""


def _is_open_access_crossref(item: dict) -> bool: # weryfikacja sposobu licencji publikacji z odpowiedzi API Crossref
    licenses = item.get("license") or []
    for lic in licenses:
        url = (lic.get("URL") or "").lower()
        if "creativecommons.org" in url or "cc-by" in url or "cc0" in url:
            return True

    links = item.get("link") or []
    for lnk in links:
        url = (lnk.get("URL") or "").lower()
        if "creativecommons.org" in url:
            return True

    return False


def _candidate_urls_from_crossref(item: dict, doi: str) -> List[str]: # łączenie informacji o DOI / URL w celu wypełniania braków
    urls = []
    if doi:
        urls.append(f"https://doi.org/{doi}")

    u = item.get("URL")
    if u:
        urls.append(str(u))

    for lnk in (item.get("link") or []):
        u2 = lnk.get("URL")
        if u2:
            urls.append(str(u2))

    # wartości unikalne z zachowaną kolejnością
    seen = set()
    out = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ----------------------------
# Crossref: pobieranie artykułów - główna funkcja
# ----------------------------
def _fetch_crossref_for_journal(
    journal_name: str,
    from_date: str,
    until_date: str,
    mailto: str = "bot@localhost",
) -> List[Dict]:
    cursor = "*"
    out: List[Dict] = []

    while True:
        params = {
            "query.container-title": journal_name,
            "filter": f"from-pub-date:{from_date},until-pub-date:{until_date},type:journal-article",
            "rows": MAX_ROWS_CROSSREF,
            "cursor": cursor,
            "mailto": mailto,  # nic nie jest wysyłane; to parametr "grzecznościowy" Crossref
        }

        r = requests.get(CROSSREF_WORKS_URL, params=params, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        message = data.get("message", {})
        items = message.get("items", []) or []
        if not items:
            break

        for item in items:
            title = _extract_title(item)
            doi = _extract_doi(item)
            abstract = _strip_tags(item.get("abstract") or "")
            status = "OpenAccess" if _is_open_access_crossref(item) else "NoOpenAccess"
            pub_date = _extract_publication_date(item) # nowe pole

            out.append({
                "title": title,
                "doi": doi,
                "journal": journal_name,
                "status": status,
                "abstract": abstract,
                "publication_date": pub_date,
                "_urls": _candidate_urls_from_crossref(item, doi),
            })

        next_cursor = message.get("next-cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(SLEEP_CROSSREF_S)

    return out


# ----------------------------
# Semantic Scholar: abstrakt po DOI - opcjonalna funkcja (Patrz - konfiguracja)
# ----------------------------
def _fetch_abstract_from_semantic_scholar(doi: str) -> Optional[str]:
    doi = _normalize_doi(doi)
    if not doi:
        return None

    url = f"{SEMANTIC_SCHOLAR_URL}/DOI:{doi}"
    params = {"fields": "abstract"}

    try:
        r = requests.get(url, params=params, timeout=(8, 12))
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        abstract = (data.get("abstract") or "").strip()
        return abstract if abstract else None
    except Exception:
        return None


# ----------------------------
# OpenAlex: abstrakt po DOI (abstract_inverted_index) - opcjonalna funkcja (Patrz - konfiguracja)
# inverted - oszczędza miejsce, może wychwycić kolejne zmienne w artykułach, agreguje wiele źródeł
# ----------------------------
def _openalex_reconstruct_abstract(abstract_inverted_index: dict) -> str:
    if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
        return ""

    max_pos = -1
    for positions in abstract_inverted_index.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""

    words = [""] * (max_pos + 1)
    for token, positions in abstract_inverted_index.items():
        for p in positions:
            if 0 <= p <= max_pos:
                words[p] = token

    text = " ".join([w for w in words if w])
    return re.sub(r"\s+", " ", text).strip()


def _fetch_from_openalex(doi: str, mailto: str = "bot@localhost") -> Tuple[Optional[str], Optional[bool]]:
    doi = _normalize_doi(doi)
    if not doi:
        return (None, None)

    work_id = f"https://doi.org/{doi}"
    url = f"{OPENALEX_WORKS_URL}/{work_id}"
    params = {"mailto": mailto}

    try:
        r = requests.get(url, params=params, timeout=(8, 15))
        if r.status_code == 404:
            return (None, None)
        r.raise_for_status()
        data = r.json()

        abs_idx = data.get("abstract_inverted_index")
        abstract = _openalex_reconstruct_abstract(abs_idx)

        oa = data.get("open_access") or {}
        is_oa = oa.get("is_oa")
        if not isinstance(is_oa, bool):
            is_oa = None

        return (abstract if abstract else None, is_oa)
    except Exception:
        return (None, None)


# ----------------------------
# HTML fallback: pozyskiwanie informacji z abstraktu z meta danych - tagów
# najbardziej odporna metoda, ale najbardziej czasochłonna
# ----------------------------
_META_PATTERNS = [
    r'<meta[^>]+name=["\']citation_abstract["\'][^>]+content=["\'](.*?)["\']',
    r'<meta[^>]+name=["\']dc\.description["\'][^>]+content=["\'](.*?)["\']',
    r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
    r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
]

def _extract_abstract_from_html(html_text: str) -> Optional[str]:
    if not html_text:
        return None

    for pat in _META_PATTERNS:
        m = re.search(pat, html_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            candidate = html_lib.unescape(m.group(1))
            candidate = re.sub(r"\s+", " ", candidate).strip()
            # prosty filtr jakości (żeby nie brać "This paper presents..." z og:description, jeśli jest krótkie)
            if len(candidate) >= 120:
                return candidate
    return None


def _html_fallback_worker(urls: List[str], q: mp.Queue) -> None:
    """
    Worker w osobnym procesie (można go zabić, jeśli się zawiesi np. na DNS).
    Zwraca abstrakt lub None przez kolejkę.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; JournalRecommender/1.0; +https://example.local)"
    }

    for u in urls[:HTML_MAX_URLS]:
        try:
            r = requests.get(u, headers=headers, timeout=(8, 12), allow_redirects=True)
            if r.status_code >= 400:
                continue

            text = r.text
            if not text or len(text) < 200:
                continue

            abs_html = _extract_abstract_from_html(text)
            if abs_html:
                q.put(abs_html)
                return
        except Exception:
            continue

    q.put(None)


def _fetch_abstract_via_html_hard_timeout(urls: List[str], hard_timeout_s: int) -> Optional[str]:
    if not urls:
        return None

    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_html_fallback_worker, args=(urls, q))
    p.daemon = True
    p.start()
    p.join(hard_timeout_s)

    if p.is_alive():
        # twarde ucięcie - przerwanie (dla uniknięcia zablokowania procesu)
        p.terminate()
        p.join(2)
        return None

    try:
        return q.get_nowait()
    except Exception:
        return None


# ----------------------------
# Uzupełnienie informacji ze wszystkich powyższych źródeł
# ----------------------------
def _enrich_records_with_more_abstracts(
    records: List[Dict],
    mailto: str = "bot@localhost",
    use_semantic_scholar: bool = True,
    use_openalex: bool = True,
    use_html_fallback: bool = True,
) -> List[Dict]:
    missing_before = sum(1 for r in records if not (r.get("abstract") or "").strip())
    print(f"\nWzbogacanie abstraktów. Brakujących przed: {missing_before}/{len(records)}\n")

    enriched = 0
    for i, rec in enumerate(records, start=1):
        if (rec.get("abstract") or "").strip():
            continue

        start_t = time.time()
        doi = _normalize_doi(rec.get("doi") or "")
        urls = rec.get("_urls") or []

        # 1) Semantic Scholar
        if use_semantic_scholar and doi and (time.time() - start_t) < PER_RECORD_BUDGET_S:
            a = _fetch_abstract_from_semantic_scholar(doi)
            time.sleep(SLEEP_S2_S)
            if a:
                rec["abstract"] = a.strip()
                enriched += 1
                continue

        # 2) OpenAlex (+ ewentualna korekta OA)
        if use_openalex and doi and (time.time() - start_t) < PER_RECORD_BUDGET_S:
            a2, is_oa = _fetch_from_openalex(doi, mailto=mailto)
            time.sleep(SLEEP_OPENALEX_S)
            if isinstance(is_oa, bool):
                rec["status"] = "OpenAccess" if is_oa else "NoOpenAccess"
            if a2:
                rec["abstract"] = a2.strip()
                enriched += 1
                continue

        # 3) HTML fallback (twardy timeout procesu)
        if use_html_fallback and urls and (time.time() - start_t) < PER_RECORD_BUDGET_S:
            a3 = _fetch_abstract_via_html_hard_timeout(urls, hard_timeout_s=HTML_FALLBACK_HARD_TIMEOUT_S)
            if a3:
                rec["abstract"] = a3.strip()
                enriched += 1
                continue

        if i % 25 == 0:
            missing_now = sum(1 for r in records if not (r.get("abstract") or "").strip())
            print(f"  Postęp: {i}/{len(records)} | uzupełniono: {enriched} | brakujących teraz: {missing_now}")

    missing_after = sum(1 for r in records if not (r.get("abstract") or "").strip())
    print(f"\nUzupełniono abstraktów: {enriched}")
    print(f"Brakujących po: {missing_after}/{len(records)}\n")
    return records


# ----------------------------
# Główna funkcja
# ----------------------------
def download_information_about_journals(
    journals_csv_path: str = os.path.join("data", "journals.csv"),
    out_csv_path: str = os.path.join("data", "list_of_data.csv"),
    from_date: str = "2025-10-01",
    until_date: str = "2025-12-31",
    mailto: str = "bot@localhost",
    dedupe: bool = True,
) -> None:
    print("=" * 70)
    print("DOWNLOAD_INFORMATION_ABOUT_JOURNALS -> list_of_data.csv")
    print("=" * 70)
    print(f"Zakres dat: {from_date} → {until_date}")
    print(f"Źródło journals.csv: {journals_csv_path}")
    print(f"Wyjście: {out_csv_path}")
    print(f"S2: {USE_SEMANTIC_SCHOLAR} | OpenAlex: {USE_OPENALEX} | HTML fallback: {USE_HTML_FALLBACK}")
    print(f"Per-record budget: {PER_RECORD_BUDGET_S}s | HTML hard-timeout: {HTML_FALLBACK_HARD_TIMEOUT_S}s")
    print()

    # Zawsze świeży plik
    if os.path.exists(out_csv_path):
        os.remove(out_csv_path)
        print(f"Usunięto stary plik: {out_csv_path}\n")

    jdf = pd.read_csv(journals_csv_path)
    if "journal" not in jdf.columns:
        raise ValueError("Brak kolumny 'journal' w journals.csv")

    journals = (
        jdf["journal"].astype(str).str.strip()
        .replace("", pd.NA).dropna().unique().tolist()
    )

    all_rows: List[Dict] = []

    # Crossref
    for idx, journal in enumerate(journals, start=1):
        print(f"[{idx}/{len(journals)}] Crossref: {journal}")
        rows = _fetch_crossref_for_journal(
            journal_name=journal,
            from_date=from_date,
            until_date=until_date,
            mailto=mailto,
        )
        print(f"Pobrano rekordów: {len(rows)}\n")
        all_rows.extend(rows)

    if not all_rows:
        print("Brak danych do zapisania.")
        return

    # Podsumowanie
    if dedupe:
        seen = set()
        deduped = []
        for r in all_rows:
            doi = (r.get("doi") or "").strip().lower()
            if doi:
                key = ("doi", doi)
            else:
                key = ("tj", (r.get("title") or "").strip().lower(), (r.get("journal") or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        all_rows = deduped
        print(f"Po deduplikacji: {len(all_rows)} rekordów\n")

    # Wzbogacenie abstracts
    all_rows = _enrich_records_with_more_abstracts(
        all_rows,
        mailto=mailto,
        use_semantic_scholar=USE_SEMANTIC_SCHOLAR,
        use_openalex=USE_OPENALEX,
        use_html_fallback=USE_HTML_FALLBACK,
    )

    # DataFrame
    columns = ["title", "doi", "journal", "status", "abstract", "publication_date"] # dodano publication_date
    df = pd.DataFrame(all_rows)

    for col in columns:
        if col not in df.columns:
            df[col] = ""

    df = df[columns].copy()
    for c in columns:
        df[c] = df[c].fillna("").astype(str).str.strip()

    df = df[df["title"] != ""].reset_index(drop=True)

    # Statystyki
    has_abs = int((df["abstract"] != "").sum())
    oa = int((df["status"] == "OpenAccess").sum())
    noa = int((df["status"] == "NoOpenAccess").sum())

    df.to_csv(out_csv_path, index=False, encoding="utf-8")

    print("=" * 70)
    print("ZAKOŃCZONO")
    print("=" * 70)
    print(f"Łącznie rekordów: {len(df)}")
    print(f"  • Z abstraktem: {has_abs} ({(100.0 * has_abs / len(df)) if len(df) else 0:.1f}%)")
    print(f"  • OpenAccess: {oa}")
    print(f"  • NoOpenAccess: {noa}")
    print(f"Zapisano do: {out_csv_path}")

# ustawienie zmiennych:
# - okres pobrania danych (w celu zwiększenia skuteczności, polecane jest pobranie większą ilość plików z mniejszymi okresami)
# - czasopisma (znajdujące się w journals.csv)
if __name__ == "__main__": # wynik pracy procesu - zebranie informacji do pliku
    download_information_about_journals(
        journals_csv_path=os.path.join("data", "journals.csv"),
        out_csv_path=os.path.join("data", "list_of_data_2026_Q1Q2.csv"),
        from_date="2026-01-01",
        until_date="2026-06-30",
        mailto="bot@localhost",
        dedupe=True,
    )