"""
crawl_club_metadata.py
Lấy metadata cơ bản của 20 CLB EPL từ Wikipedia infobox.
Fields: stadium, city, manager, founded, capacity, nickname, chairman, league

Cài: pip install requests
Chạy: python crawl_club_metadata.py
Output: ./epl_transfers/metadata_clubs.json
         ./epl_transfers/metadata_clubs.txt   ← RAG-ready
"""

import random

import requests, json, re, time, os

OUTPUT_DIR  = "./epl_transfers"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "metadata_clubs.json")
OUTPUT_TXT  = os.path.join(OUTPUT_DIR, "metadata_clubs.txt")

HEADERS = {
    "User-Agent": "EPL-RAG-Research/1.0 (student NLP project; python-requests; contact: 23020439@vnu.edu.vn)"
}

# Slug Wikipedia chính xác cho từng CLB
EPL_CLUBS_WIKI = [
    ("Manchester City",   "Manchester_City_F.C.",          1),
    ("Arsenal",           "Arsenal_F.C.",                   2),
    ("Liverpool",         "Liverpool_F.C.",                 3),
    ("Aston Villa",       "Aston_Villa_F.C.",               4),
    ("Tottenham Hotspur", "Tottenham_Hotspur_F.C.",         5),
    ("Chelsea",           "Chelsea_F.C.",                   6),
    ("Newcastle United",  "Newcastle_United_F.C.",          7),
    ("Manchester United", "Manchester_United_F.C.",         8),
    ("West Ham United",   "West_Ham_United_F.C.",           9),
    ("Crystal Palace",    "Crystal_Palace_F.C.",           10),
    ("Brighton",          "Brighton_&_Hove_Albion_F.C.",   11),
    ("Wolverhampton",     "Wolverhampton_Wanderers_F.C.",  12),
    ("Fulham",            "Fulham_F.C.",                   13),
    ("Bournemouth",       "AFC_Bournemouth",               14),
    ("Nottingham Forest", "Nottingham_Forest_F.C.",        15),
    ("Everton",           "Everton_F.C.",                  16),
    ("Brentford",         "Brentford_F.C.",                17),
    ("Luton Town",        "Luton_Town_F.C.",               18),
    ("Burnley",           "Burnley_F.C.",                  19),
    ("Sheffield United",  "Sheffield_United_F.C.",         20),
]

# Fallback city/location mapping vì nhiều Wikipedia club infobox không có field city/location
CITY_FALLBACK = {
    "Manchester City": "Manchester",
    "Arsenal": "London",
    "Liverpool": "Liverpool",
    "Aston Villa": "Birmingham",
    "Tottenham Hotspur": "London",
    "Chelsea": "London",
    "Newcastle United": "Newcastle upon Tyne",
    "Manchester United": "Manchester",
    "West Ham United": "London",
    "Crystal Palace": "London",
    "Brighton": "Brighton and Hove",
    "Wolverhampton": "Wolverhampton",
    "Fulham": "London",
    "Bournemouth": "Bournemouth",
    "Nottingham Forest": "Nottingham",
    "Everton": "Liverpool",
    "Brentford": "London",
    "Luton Town": "Luton",
    "Burnley": "Burnley",
    "Sheffield United": "Sheffield",
}

# Nickname phổ biến nhất cho mỗi CLB
PRIMARY_NICKNAME = {
    "Manchester City": "The Citizens",
    "Arsenal": "The Gunners",
    "Liverpool": "The Reds",
    "Aston Villa": "The Villans",
    "Tottenham Hotspur": "The Lilywhites",
    "Chelsea": "The Blues",
    "Newcastle United": "The Magpies",
    "Manchester United": "The Red Devils",
    "West Ham United": "The Hammers",
    "Crystal Palace": "The Eagles",
    "Brighton": "The Seagulls",
    "Wolverhampton": "The Wolves",
    "Fulham": "The Cottagers",
    "Bournemouth": "The Cherries",
    "Nottingham Forest": "Forest",
    "Everton": "The Toffees",
    "Brentford": "The Bees",
    "Luton Town": "The Hatters",
    "Burnley": "The Clarets",
    "Sheffield United": "The Blades",
}


# Safe GET with retry and exponential backoff
def safe_get(url: str, params: dict, retries: int = 8):
    """HTTP GET với retry + exponential backoff để tránh 429."""
    for attempt in range(retries):
        try:
            r = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=(10, 45),
            )

            # Wikipedia rate limit
            if r.status_code == 429:
                wait = min(30, 2 ** attempt) + random.uniform(1, 3)
                print(f"    [429] Rate limited — sleeping {wait:.1f}s")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r

        except requests.RequestException as e:
            if attempt == retries - 1:
                raise e

            wait = min(30, 2 ** attempt) + random.uniform(1, 3)
            print(f"    [RETRY] {e} — sleeping {wait:.1f}s")
            time.sleep(wait)

    return None


def fetch_wikitext(title: str) -> str | None:
    """Lấy raw wikitext của bài Wikipedia qua Action API."""
    params = {
        "action":  "query",
        "format":  "json",
        "prop":    "revisions",
        "titles":  title.replace("_", " "),
        "rvprop":  "content",
        "rvslots": "main",
    }
    try:
        r = safe_get(
            "https://en.wikipedia.org/w/api.php",
            params=params
        )
        # r.raise_for_status()
    
        if r is None:
            return None

        pages = r.json().get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid == "-1":
                return None
            revs = page.get("revisions", [])
            if revs:
                return revs[0].get("slots", {}).get("main", {}).get("*", "")
        return None
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


def parse_infobox_field(wikitext: str, *field_names: str) -> str:
    """
    Trích xuất giá trị của 1 field trong {{Infobox ...}}.
    Hỗ trợ nhiều tên field (alias).
    """
    lines = wikitext.splitlines()

    for field in field_names:
        # Pattern: | field = value (value có thể multiline, chứa wikilinks)
        # pattern = rf"\|\s*{re.escape(field)}\s*=\s*([^\|\}}]+)"
        # m = re.search(pattern, wikitext, re.IGNORECASE)
        # if m:
        #     raw = m.group(1).strip()
        field_pattern = re.compile(
            rf"^\|\s*{re.escape(field)}\s*=\s*(.*)$",
            re.IGNORECASE,
        )

        collecting = False
        value_lines = []

        for line in lines:
            # Bắt đầu field
            if not collecting:
                m = field_pattern.match(line)
                if m:
                    collecting = True
                    value_lines.append(m.group(1).strip())
                continue

            # Sang field mới thì dừng
            if re.match(r"^\|\s*[a-zA-Z0-9_]+\s*=", line):
                break

            # Tiếp tục multiline value
            value_lines.append(line.strip())

        if value_lines:
            raw = " ".join(v for v in value_lines if v)
            return clean_wiki_value(raw)
    return ""


def clean_wiki_value(raw: str) -> str:
    """Làm sạch raw Wikipedia markup."""

    if not raw:
        return ""

    # Remove refs first
    raw = re.sub(r"<ref[^>]*>.*?</ref>", "", raw, flags=re.S | re.I)
    raw = re.sub(r"<ref[^/]*/>", "", raw, flags=re.I)

    # {{start date and age|1880|...}} -> 1880
    raw = re.sub(
        r"\{\{\s*start date(?: and age)?\s*\|(?:df=yes\|)?\s*(\d{4})[^}]*\}\}",
        r"\1",
        raw,
        flags=re.I,
    )

    # Chỉ giữ lại năm đầu tiên nếu field founded chứa nhiều mốc lịch sử
    # Ví dụ:
    # start date and age|df=yes|1880 as St. Mark's
    # -> 1880
    year_match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", raw)
    if year_match:
        raw = year_match.group(1)

    # Remove refs like <ref>...</ref>
    # raw = re.sub(r"<ref[^>]*>.*?</ref>", "", raw, flags=re.S)
    # {{plainlist|...}}, {{ubl|...}}, {{hlist|...}}
    raw = re.sub(
        r"\{\{(?:plainlist|ubl|hlist|unbulleted list)\|",
        "",
        raw,
        flags=re.I,
    )

    # # [[link|display]] → display
    # raw = re.sub(r"\[\[(?:[^\|\]]+\|)?([^\]]+)\]\]", r"\1", raw)
    # # {{plainlist|...}} hoặc {{ubl|...}} → lấy nội dung
    # raw = re.sub(r"\{\{(?:plainlist|ubl|hlist|unbulleted list)[^}]*\}\}", "", raw, flags=re.I)
    # # {{nowrap|text}} → text
    # raw = re.sub(r"\{\{nowrap\|([^}]+)\}\}", r"\1", raw, flags=re.I)
    
    # Remove remaining template braces
    raw = raw.replace("{{", " ").replace("}}", " ")

    # [[A|B]] -> B
    raw = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", raw)

    # External links
    raw = re.sub(r"\[https?://[^\s]+\s([^\]]+)\]", r"\1", raw)

    # Bỏ templates còn lại
    raw = re.sub(r"\{\{[^}]*\}\}", "", raw)
    
    # Bỏ HTML tags
    raw = re.sub(r"<[^>]+>", "", raw)
    # # Bỏ wiki markup
    # raw = re.sub(r"'''?", "", raw)
    
    # Remove wiki bold/italic
    raw = re.sub(r"'{2,}", "", raw)

    # Convert list separators
    raw = raw.replace("*", ", ")

    # Normalize separators
    # raw = raw.replace("|", ", ")
    
    # Normalize whitespace
    raw = re.sub(r"\s+", " ", raw).strip()
    # # Bỏ trailing comma/semicolon
    # raw = raw.strip(",.;")
    
    # Cleanup repeated commas
    raw = re.sub(r",\s*,+", ", ", raw)

    # Remove trailing punctuation
    raw = raw.strip(" ,.;")
    
    return raw


def extract_club_metadata(wikitext: str, club_name: str, position: int) -> dict:
    """Parse {{Infobox football club}} fields."""
    meta = {
        "club":            club_name,
        "final_position_2023_24": position,
        "season":          "2023-24",
        "stadium":         parse_infobox_field(wikitext, "ground", "stadium"),
        "city":            parse_infobox_field(wikitext, "city", "location", "ground_location") or CITY_FALLBACK.get(club_name, ""),
        "manager":         parse_infobox_field(wikitext, "manager", "mgr", "head_coach"),
        "founded":         parse_infobox_field(wikitext, "founded", "established", "formation"),
        "capacity":        parse_infobox_field(wikitext, "capacity"),
        "nickname":        PRIMARY_NICKNAME.get(club_name, parse_infobox_field(wikitext, "nickname", "nicknames")),
        "chairman":        parse_infobox_field(wikitext, "chairman", "owner", "president"),
        "league":          parse_infobox_field(wikitext, "league", "division"),
        "kit_colours":     parse_infobox_field(wikitext, "kit_body", "body_colour"),
    }
    # Làm sạch capacity — chỉ lấy số
    cap = re.sub(r"[^\d,]", "", meta["capacity"])
    if cap:
        meta["capacity"] = cap

    # Lọc bỏ field trống
    return {k: v for k, v in meta.items() if v}


def to_text_block(meta: dict) -> str:
    """Chuyển metadata thành đoạn văn bản cho RAG."""
    c = meta["club"]
    lines = [
        f"## {c} — Club Profile",
        f"Season: 2023-24 Premier League (Final position: {meta.get('final_position_2023_24', '?')}/20)",
        "",
    ]
    field_labels = {
        "stadium":   "Stadium",
        "city":      "City / Location",
        "manager":   "Manager (2023-24)",
        "founded":   "Year Founded",
        "capacity":  "Stadium Capacity",
        "nickname":  "Nickname(s)",
        "chairman":  "Chairman / Owner",
        "league":    "League",
    }
    for key, label in field_labels.items():
        if meta.get(key):
            lines.append(f"  {label}: {meta[key]}")
    return "\n".join(lines)


# ════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════

def main():
    print("═"*58)
    print("  EPL CLUB METADATA CRAWLER — Wikipedia Infoboxes")
    print("═"*58)

    all_meta = []
    failed   = []

    for club_name, wiki_slug, position in EPL_CLUBS_WIKI:
        print(f"\n[{position:02d}/20] {club_name}")
        wikitext = fetch_wikitext(wiki_slug)

        if not wikitext:
            print(f"       [SKIP] Not found on Wikipedia")
            failed.append(club_name)
            continue

        meta = extract_club_metadata(wikitext, club_name, position)

        # Hiển thị kết quả
        print(f"       Stadium  : {meta.get('stadium', '—')}")
        print(f"       City     : {meta.get('city', '—')}")
        print(f"       Manager  : {meta.get('manager', '—')}")
        print(f"       Founded  : {meta.get('founded', '—')}")
        print(f"       Capacity : {meta.get('capacity', '—')}")
        print(f"       Nickname : {meta.get('nickname', '—')}")

        all_meta.append(meta)
                # polite rate limiting + jitter
        time.sleep(random.uniform(1.5, 3.0))

    # ── Lưu JSON ────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    print(f"\n✓ JSON saved: {OUTPUT_JSON}")

    # ── Lưu TXT cho RAG ─────────────────────────────────
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("# EPL 2023-24 CLUB PROFILES — Metadata\n\n")
        for meta in all_meta:
            f.write(to_text_block(meta))
            f.write("\n\n" + "─"*50 + "\n\n")
    print(f"✓ TXT  saved: {OUTPUT_TXT}")

    if failed:
        print(f"\n[WARN] Failed: {failed}")

    print(f"\n{'═'*58}")
    print(f"  Done — {len(all_meta)}/20 clubs crawled")
    print(f"{'═'*58}")


if __name__ == "__main__":
    main()
