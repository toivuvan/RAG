"""
╔══════════════════════════════════════════════════════════════════╗
║   EPL 2023-24 TRANSFER CRAWLER — salarysport.com               ║
║   Dữ liệu: tên cầu thủ | phí | CLB đi/đến | vị trí | tuổi    ║
╠══════════════════════════════════════════════════════════════════╣
║  Cài đặt:  pip install requests beautifulsoup4 lxml             ║
║  Chạy:     python crawl_epl_transfers.py                        ║
║  Output:   ./epl_transfers/ (20 .txt + 2 .csv + 1 .json)       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import requests
from bs4 import BeautifulSoup
import time, os, csv, json, re

OUTPUT_DIR = "./epl_transfers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://salarysport.com/football/",
}

# Slug lấy trực tiếp từ links trên trang Arsenal của salarysport.com
EPL_CLUBS = [
    ("Manchester City",   "manchester-city-f.c.",          1),
    ("Arsenal",           "arsenal-f.c.",                   2),
    ("Liverpool",         "liverpool-f.c.",                 3),
    ("Aston Villa",       "aston-villa-f.c.",               4),
    ("Tottenham Hotspur", "tottenham-hotspur-f.c.",         5),
    ("Chelsea",           "chelsea-f.c.",                   6),
    ("Newcastle United",  "newcastle-united-f.c.",          7),
    ("Manchester United", "manchester-united-f.c.",         8),
    ("West Ham United",   "west-ham-united-f.c.",           9),
    ("Crystal Palace",    "crystal-palace",                10),
    ("Brighton",          "brighton-&-hove-albion",        11),
    ("Wolverhampton",     "wolverhampton-wanderers-f.c.",  12),
    ("Fulham",            "fulham",                        13),
    ("Bournemouth",       "afc-bournemouth",               14),
    ("Nottingham Forest", "nottingham-forest",             15),
    ("Everton",           "everton-f.c.",                  16),
    ("Brentford",         "brentford",                     17),
    ("Luton Town",        "luton-town",                    18),
    ("Burnley",           "burnley-f.c.",                  19),
    ("Sheffield United",  "sheffield-united-f.c.",         20),
]

BASE_URL = "https://salarysport.com/football/{slug}/transfers/"


def fetch(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        # Force UTF-8: requests đoán sai charset dẫn đến é → Ã©, í → Ã­
        r.encoding = "utf-8"
        return BeautifulSoup(r.text, "lxml")
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


def parse_table(tbl) -> list[dict]:
    thead = tbl.find("thead")
    if not thead:
        return []
    cols = [th.get_text(strip=True) for th in thead.find_all("th")]
    rows = []
    for tr in (tbl.find("tbody") or tbl).find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells and len(cells) == len(cols):
            rows.append(dict(zip(cols, cells)))
    return rows


def extract_transfers(soup, club_name):
    """
    Lấy arrivals & departures của mùa 2023/24.
    Chiến lược: tìm heading có '2023/24', rồi thu thập 
    các bảng ngay bên dưới cho đến khi gặp heading mùa khác.
    """
    arrivals, departures = [], []
    if not soup:
        return arrivals, departures

    body = soup.find("body")
    elems = body.find_all(["h1","h2","h3","h4","p","table"]) if body else []

    in_2324 = False
    direction = None

    for el in elems:
        tag = el.name
        text = el.get_text(" ", strip=True) if tag != "table" else ""

        if tag in ("h1","h2","h3","h4","p"):
            # Phát hiện mùa
            if "2023/24" in text or "2023-24" in text:
                in_2324 = True
            elif in_2324 and re.search(r"202[0-2]/2[0-4]|2019|2020|2021|2022", text):
                in_2324 = False   # qua mùa khác, dừng

            if in_2324:
                low = text.lower()
                if any(w in low for w in ("arrival","transfer in","signing","ins")):
                    direction = "arrivals"
                elif any(w in low for w in ("departure","transfer out","sale","outs")):
                    direction = "departures"

        elif tag == "table" and in_2324 and direction:
            for row in parse_table(el):
                row["club"]      = club_name
                row["season"]    = "2023-24"
                row["direction"] = direction
                if direction == "arrivals":
                    arrivals.append(row)
                else:
                    departures.append(row)

    return arrivals, departures


def summary_from_page(soup):
    """Lấy tổng chi tiêu / thu về của mùa 2023/24 từ text."""
    lines = []
    for tag in soup.find_all(["h2","h3","p"]):
        t = tag.get_text(" ", strip=True)
        if ("2023/24" in t or "2023-24" in t) and any(
            w in t.lower() for w in ("spending","net","balance","total","sale","purchase")
        ):
            lines.append(t)
    return " | ".join(lines[:4])


def to_text(club_name, position, summary, arrivals, departures):
    lines = [
        f"# {club_name} — EPL 2023-24 Transfer Window",
        f"Final league position: {position}/20 (2023-24 Premier League)",
        "",
    ]
    if summary:
        lines += ["## Financial Summary", f"  {summary}", ""]

    lines.append(f"## Players Signed (Arrivals) — {len(arrivals)} transfers")
    if arrivals:
        for r in arrivals:
            player = r.get("Player Name", r.get("Player","?"))
            fee    = r.get("Transfer Fee", r.get("Fee","?"))
            src    = r.get("From", r.get("Club","?"))
            league = r.get("League","")
            pos    = r.get("Position","")
            age    = r.get("Age","")
            nat    = r.get("Nationality","")
            line   = f"  - {player} signed from {src}"
            if league: line += f" ({league})"
            line  += f" | Fee: {fee}"
            if pos: line += f" | Pos: {pos}"
            if age: line += f" | Age: {age}"
            if nat: line += f" | Nat: {nat}"
            lines.append(line)
    else:
        lines.append("  (no data)")

    lines += ["", f"## Players Sold / Departed — {len(departures)} transfers"]
    if departures:
        for r in departures:
            player = r.get("Player Name", r.get("Player","?"))
            fee    = r.get("Transfer Fee", r.get("Fee","?"))
            dst    = r.get("To", r.get("Club","?"))
            league = r.get("League","")
            pos    = r.get("Position","")
            age    = r.get("Age","")
            nat    = r.get("Nationality","")
            line   = f"  - {player} sold/moved to {dst}"
            if league: line += f" ({league})"
            line  += f" | Fee: {fee}"
            if pos: line += f" | Pos: {pos}"
            if age: line += f" | Age: {age}"
            if nat: line += f" | Nat: {nat}"
            lines.append(line)
    else:
        lines.append("  (no data)")

    return "\n".join(lines)


# ════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════
print("═"*62)
print("  EPL 2023-24 TRANSFER DATA CRAWLER")
print("  Source: salarysport.com")
print("═"*62)

all_arr, all_dep, results = [], [], []

for club_name, slug, pos in EPL_CLUBS:
    url  = BASE_URL.format(slug=slug)
    print(f"\n[{pos:02d}/20] {club_name}")

    soup = fetch(url)
    arr, dep = extract_transfers(soup, club_name)
    summ = summary_from_page(soup) if soup else ""

    print(f"       arrivals={len(arr)}  departures={len(dep)}")

    # Lưu txt (1 file per club)
    txt      = to_text(club_name, pos, summ, arr, dep)
    safe     = club_name.lower().replace(" ","_")
    fname    = f"{pos:02d}_{safe}_2023_24.txt"
    fpath    = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"       → {fname}")

    all_arr.extend(arr)
    all_dep.extend(dep)
    results.append({
        "club": club_name, "position": pos, "url": url,
        "summary": summ, "arrivals": arr, "departures": dep,
    })
    time.sleep(1.5)


# ── Save combined outputs ────────────────────────────────────────
print("\n\nSaving combined files...")

def save_csv(rows, path):
    if not rows: return
    cols = list({k for r in rows for k in r})
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

save_csv(all_arr, os.path.join(OUTPUT_DIR, "all_arrivals_2023_24.csv"))
save_csv(all_dep, os.path.join(OUTPUT_DIR, "all_departures_2023_24.csv"))

with open(os.path.join(OUTPUT_DIR,"all_clubs_2023_24.json"),"w",encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# File RAG tổng hợp (toàn bộ 20 CLB trong 1 file lớn)
with open(os.path.join(OUTPUT_DIR,"EPL_ALL_TRANSFERS_2023_24.txt"),"w",encoding="utf-8") as f:
    f.write("# ENGLISH PREMIER LEAGUE 2023-24 — COMPLETE TRANSFER DATA\n")
    f.write("# All 20 clubs, Summer 2023 + Winter 2024 windows\n")
    f.write("# Source: salarysport.com\n\n")
    for r in results:
        f.write(to_text(r["club"], r["position"], r["summary"], r["arrivals"], r["departures"]))
        f.write("\n\n" + "─"*60 + "\n\n")

print(f"""
{'═'*62}
DONE ✓
  Clubs crawled  : {len(results)} / 20
  Total arrivals : {len(all_arr)}
  Total departures: {len(all_dep)}
  Output folder  : {os.path.abspath(OUTPUT_DIR)}/
{'═'*62}

Files:
  {len(results)} × club_XX_name_2023_24.txt    ← per-club RAG docs
  all_arrivals_2023_24.csv           ← bảng tổng hợp INS
  all_departures_2023_24.csv         ← bảng tổng hợp OUTS
  all_clubs_2023_24.json             ← raw JSON đầy đủ
  EPL_ALL_TRANSFERS_2023_24.txt      ← 1 file RAG tổng hợp
""")