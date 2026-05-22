import json
import math
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

def _fmt(val):
    """Chuyển giá trị rỗng/NaN thành chuỗi 'N/A'."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return str(val)

def _normalize_id(value: str) -> str:
    """Chuẩn hóa tên thành id ngắn, loại bỏ dấu, viết thường, thay space bằng gạch dưới."""
    return (
        _fmt(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
        .replace('"', "")
    )

def _row_text(prefix: str, row: pd.Series) -> str:
    """Biến một dòng CSV thành text nhiều dòng để đưa vào chunk."""
    lines = [f"# {prefix}"]
    for key, val in row.items():
        lines.append(f"  {key}: {_fmt(val)}")
    return "\n".join(lines)


def _load_csv_row_chunks(path: Path, source_name: str, pos_filter=None) -> list[dict]:
    """Đọc một CSV bất kỳ và tạo chunk theo từng dòng, có thể lọc theo vị trí."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if pos_filter is not None and "pos" in df.columns:
        df = df[df["pos"].astype(str).apply(pos_filter)]

    chunks = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        player = row_dict.get("player") or row_dict.get("Player Name") or "unknown"
        team = row_dict.get("team") or row_dict.get("club") or "unknown"
        chunk_id = f"{source_name}_{_normalize_id(team)}_{_normalize_id(player)}_{idx}"
        chunks.append({
            "id": chunk_id,
            "text": _row_text(f"{source_name.replace('_', ' ').title()} - {player}", row),
            "source": source_name,
            "team": team if team != "unknown" else None,
            "metadata": {"source_file": path.name, **{k: _fmt(v) for k, v in row_dict.items()}},
        })
    return chunks


def _load_transfer_chunks_from_csv(path: Path, direction: str) -> list[dict]:
    """Tạo chunk chuyển nhượng từ file arrivals/departures CSV."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    chunks = []
    for idx, row in df.iterrows():
        player = row.get("Player Name") or row.get("player") or "unknown"
        team = row.get("club") or row.get("team") or "unknown"
        direction_label = direction.capitalize()
        extra = []
        if direction == "arrivals":
            extra.append(f"From: {_fmt(row.get('From'))}")
            extra.append(f"To: {team}")
        else:
            extra.append(f"From: {team}")
            extra.append(f"To: {_fmt(row.get('To'))}")
        text = [f"# {direction_label} — {player}"]
        text.append(f"  Team: {team} | League: {_fmt(row.get('League'))} | Position: {_fmt(row.get('Position'))}")
        text.append(f"  Fee: {_fmt(row.get('Transfer Fee'))} | Nation: {_fmt(row.get('Nationality'))} | Age: {_fmt(row.get('Age'))}")
        text.extend([f"  {item}" for item in extra])
        chunks.append({
            "id": f"transfer_{direction}_{_normalize_id(player)}_{idx}",
            "text": "\n".join(text),
            "source": direction,
            "team": team if team != "unknown" else None,
            "metadata": {"source_file": path.name, "direction": direction, **{k: _fmt(v) for k, v in row.items()}},
        })
    return chunks


def _now_ts() -> str:
    """Trả về timestamp UTC dạng ISO để gắn vào chunk mới tạo."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _format_league_season(league: str, season: str) -> str:
    """Chuẩn hóa tên giải và mùa giải"""
    league = _fmt(league).replace("ENG-", "").strip()
    season = _fmt(season)
    if season == "2324":
        season = "2023-24"
    return f"{league} {season}".strip()


def _safe_field(row: dict, key: str) -> str:
    """Lấy field từ dict và chuẩn hóa giá trị rỗng bằng _fmt."""
    return _fmt(row.get(key, "N/A"))


def _build_player_metadata(row: dict) -> dict:
    """Tạo metadata chuẩn cho một cầu thủ từ dòng dữ liệu raw."""
    return {
        "player_name": _safe_field(row, "player"),
        "team": _safe_field(row, "team"),
        "position": _safe_field(row, "pos"),
        "age": int(row["age"]) if row.get("age") not in (None, "N/A", "") and str(row.get("age")).isdigit() else _safe_field(row, "age"),
        "nationality": _safe_field(row, "nation"),
        "league_season": _format_league_season(row.get("league", "Premier League"), row.get("season", "2324")),
    }


def _build_player_semantic_content(row: dict) -> str:
    """Tạo đoạn mô tả tự nhiên về cầu thủ để phục vụ semantic retrieval."""
    player = _safe_field(row, "player")
    team = _safe_field(row, "team")
    pos = _safe_field(row, "pos")
    nation = _safe_field(row, "nation")
    league_season = _format_league_season(row.get("league", "Premier League"), row.get("season", "2324"))
    mp = _safe_field(row, "Playing Time_MP")
    mins = _safe_field(row, "Playing Time_Min")
    goals = _safe_field(row, "Performance_Gls")
    assists = _safe_field(row, "Performance_Ast")
    tackles = _safe_field(row, "Performance_TklW")
    if tackles == "N/A":
        tackles = _safe_field(row, "TklW")
    return (
        f"{player} ({_safe_field(row, 'age')}) is a {pos} for {team} and plays for the {nation} national team. "
        f"In the {league_season} season, {player} made {mp} appearances, playing {mins} minutes. "
        f"He scored {goals} goals and provided {assists} assists. He made {tackles} tackles."
    )


def _build_player_structured_content(row: dict) -> str:
    """Tạo chunk dạng cấu trúc/markdown chứa thống kê chi tiết của cầu thủ."""
    league_season = _format_league_season(row.get("league", "Premier League"), row.get("season", "2324"))
    lines = [f"# {_safe_field(row, 'player')}", "", f"## Chi tiết Thống kê (Season {league_season}):"]
    sections = [
        ("⏱️ Thời gian thi đấu (Playing Time)", [
            ("90s", "Playing Time_90s"),
            ("MP", "Playing Time_MP"),
            ("Min", "Playing Time_Min"),
            ("Mn/MP", "Playing Time_Mn/MP"),
            ("Min%", "Playing Time_Min%"),
            ("Starts", "Starts_Starts"),
            ("Subs", "Subs_Subs"),
        ]),
        ("⚽ Thành tích (Performance)", [
            ("Gls", "Performance_Gls"),
            ("Ast", "Performance_Ast"),
            ("G+A", "Performance_G+A"),
            ("G-PK", "Performance_G-PK"),
            ("PK", "Performance_PK"),
            ("PKatt", "Performance_PKatt"),
            ("CrdY", "Performance_CrdY"),
            ("CrdR", "Performance_CrdR"),
            ("2CrdY", "Performance_2CrdY"),
            ("Fls", "Performance_Fls"),
            ("Fld", "Performance_Fld"),
            ("Off", "Performance_Off"),
            ("Crs", "Performance_Crs"),
            ("Int", "Performance_Int"),
            ("TklW", "Performance_TklW"),
            ("OG", "Performance_OG"),
            ("PKwon", "Performance_PKwon"),
            ("PKcon", "Performance_PKcon"),
        ]),
        ("📊 Chỉ số trên 90 phút (Per 90 Minutes)", [
            ("Gls", "Per 90 Minutes_Gls"),
            ("Ast", "Per 90 Minutes_Ast"),
            ("G+A", "Per 90 Minutes_G+A"),
            ("G+A-PK", "Per 90 Minutes_G+A-PK"),
            ("G-PK", "Per 90 Minutes_G-PK"),
        ]),
        ("🎯 Sút bóng (Shooting)", [
            ("Sh", "Standard_Sh"),
            ("SoT", "Standard_SoT"),
            ("SoT%", "Standard_SoT%"),
            ("Sh/90", "Standard_Sh/90"),
            ("SoT/90", "Standard_SoT/90"),
            ("G/Sh", "Standard_G/Sh"),
            ("G/SoT", "Standard_G/SoT"),
            ("Gls", "Standard_Gls"),
            ("PK", "Standard_PK"),
            ("PKatt", "Standard_PKatt"),
        ]),
        ("🛡️ Phòng ngự (Defense)", [
            ("Fls", "Performance_Fls"),
            ("Int", "Performance_Int"),
            ("TklW", "Performance_TklW"),
        ]),
    ]

    for title, fields in sections:
        lines.append("")
        lines.append(f"**{title}:**")
        for label, field in fields:
            lines.append(f"- {label}: {_safe_field(row, field)}")
    return "\n".join(lines)


def _build_player_chunks(raw_dir: Path) -> list[dict]:
    """Gộp nhiều bảng player stats và tạo semantic + structured chunk cho từng cầu thủ."""
    files = [
        raw_dir / "standard_stats.csv",
        raw_dir / "playing_time_stats.csv",
        raw_dir / "shooting_stats.csv",
        raw_dir / "misc_stats.csv",
    ]
    df = pd.read_csv(files[0])
    for path in files[1:]:
        df = df.merge(
            pd.read_csv(path),
            how="left",
            on=["league", "season", "team", "player", "nation", "pos", "age", "born"],
            suffixes=("", "_aux"),
        )

    chunks = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        doc_id = f"player_{_normalize_id(row_dict.get('player', 'unknown'))}"
        metadata = _build_player_metadata(row_dict)
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_semantic",
            "chunk_type": "semantic",
            "doc_type": "player",
            "content": _build_player_semantic_content(row_dict),
            "metadata": metadata,
            "created_at": _now_ts(),
        })
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_structured",
            "chunk_type": "structured",
            "doc_type": "player",
            "content": _build_player_structured_content(row_dict),
            "metadata": metadata,
            "created_at": _now_ts(),
        })
    return chunks


def _build_team_metadata(row: dict) -> dict:
    """Tạo metadata chuẩn cho đội bóng."""
    return {
        "team_name": _safe_field(row, "team"),
        "league_season": _format_league_season(row.get("league", "Premier League"), row.get("season", "2324")),
        "possession": _safe_field(row, "Poss"),
        "goals": _safe_field(row, "Performance_Gls"),
        "assists": _safe_field(row, "Performance_Ast"),
        "yellow_cards": _safe_field(row, "Performance_CrdY"),
        "red_cards": _safe_field(row, "Performance_CrdR"),
    }


def _build_team_semantic_content(row: dict) -> str:
    """Tạo đoạn mô tả tự nhiên về thống kê tổng hợp của đội bóng."""
    team = _safe_field(row, "team")
    league_season = _format_league_season(row.get("league", "Premier League"), row.get("season", "2324"))
    return (
        f"{team} competed in the {league_season}. They scored {_safe_field(row, 'Performance_Gls')} goals, created {_safe_field(row, 'Performance_Ast')} assists, "
        f"held {_safe_field(row, 'Poss')}% possession, and received {_safe_field(row, 'Performance_CrdY')} yellow cards and {_safe_field(row, 'Performance_CrdR')} red cards."
    )


def _build_team_structured_content(row: dict) -> str:
    """Tạo chunk markdown chứa thống kê đội bóng chi tiết."""
    lines = [f"# {_safe_field(row, 'team')}", "", "## Team Aggregate Stats (2023-24 Premier League):", ""]
    lines.append(f"- Players used: {_safe_field(row, 'players_used')}")
    lines.append(f"- Avg age: {_safe_field(row, 'Age')}")
    lines.append(f"- Possession: {_safe_field(row, 'Poss')}%")
    lines.append(f"- Goals: {_safe_field(row, 'Performance_Gls')}")
    lines.append(f"- Assists: {_safe_field(row, 'Performance_Ast')}")
    lines.append(f"- Yellow cards: {_safe_field(row, 'Performance_CrdY')}")
    lines.append(f"- Red cards: {_safe_field(row, 'Performance_CrdR')}")
    lines.append(f"- Fouls committed: {_safe_field(row, 'Performance_Fls')}")
    lines.append(f"- Fouls drawn: {_safe_field(row, 'Performance_Fld')}")
    lines.append(f"- Offsides: {_safe_field(row, 'Performance_Off')}")
    lines.append("\n## Shooting Stats:")
    lines.append(f"- Shots: {_safe_field(row, 'Standard_Sh')}")
    lines.append(f"- Shots on target: {_safe_field(row, 'Standard_SoT')}")
    lines.append(f"- Shot accuracy: {_safe_field(row, 'Standard_SoT%')}%")
    return "\n".join(lines)


def _build_team_chunks(raw_dir: Path) -> list[dict]:
    """Tạo chunk cho từng đội."""
    ts = pd.read_csv(raw_dir / "team_stats.csv")
    sh = pd.read_csv(raw_dir / "team_shooting_stats.csv")
    mis = pd.read_csv(raw_dir / "team_misc_stats.csv")
    df = ts.merge(sh, how="left", on="team", suffixes=("", "_shooting"))
    df = df.merge(mis, how="left", on="team", suffixes=("", "_misc"))

    chunks = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        doc_id = f"team_{_normalize_id(row_dict.get('team', 'unknown'))}"
        metadata = _build_team_metadata(row_dict)
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_semantic",
            "chunk_type": "semantic",
            "doc_type": "team",
            "content": _build_team_semantic_content(row_dict),
            "metadata": metadata,
            "created_at": _now_ts(),
        })
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_structured",
            "chunk_type": "structured",
            "doc_type": "team",
            "content": _build_team_structured_content(row_dict),
            "metadata": metadata,
            "created_at": _now_ts(),
        })
    return chunks


def _build_club_metadata_chunks(raw_dir: Path) -> list[dict]:
    """Tạo chunk metadata CLB."""
    meta_path = raw_dir / "metadata_clubs.json"
    if not meta_path.exists():
        return []
    with open(meta_path, encoding="utf-8") as fh:
        clubs = json.load(fh)

    chunks = []
    for club in clubs:
        team_name = club.get("club", "unknown")
        season = _fmt(club.get("season", "2023-24"))
        league_value = _format_league_season("Premier League", season)
        final_position = _fmt(club.get("final_position_2023_24", club.get("final_position", "N/A")))

        doc_id = f"team_{_normalize_id(team_name)}"
        chunk_id = f"{doc_id}_metadata"
        content_lines = [f"# {team_name}", f"- club: {team_name}", f"- final_position: {final_position}", f"- season: {season}", f"- stadium: {_fmt(club.get('stadium'))}", f"- city: {_fmt(club.get('city'))}", f"- manager: {_fmt(club.get('manager'))}", f"- founded: {_fmt(club.get('founded'))}", f"- capacity: {_fmt(club.get('capacity'))}", f"- nickname: {_fmt(club.get('nickname'))}", f"- chairman: {_fmt(club.get('chairman'))}", f"- league: {league_value}"]

        metadata = {
            "team_name": team_name,
            "source_file": meta_path.name,
            "club": team_name,
            "final_position": final_position,
            "season": season,
            "stadium": _fmt(club.get("stadium")),
            "city": _fmt(club.get("city")),
            "manager": _fmt(club.get("manager")),
            "founded": _fmt(club.get("founded")),
            "capacity": _fmt(club.get("capacity")),
            "nickname": _fmt(club.get("nickname")),
            "chairman": _fmt(club.get("chairman")),
            "league": league_value,
        }

        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_type": "metadata",
            "doc_type": "team",
            "content": "\n".join(content_lines),
            "metadata": metadata,
            "created_at": _now_ts(),
        })
    return chunks


def _build_match_metadata(row: dict) -> dict:
    """Tạo metadata chuẩn cho một trận đấu."""
    return {
        "home_team": _safe_field(row, "home_team"),
        "away_team": _safe_field(row, "away_team"),
        "venue": _safe_field(row, "venue"),
        "score": _safe_field(row, "score"),
        "date": _safe_field(row, "date"),
        "league_season": "Premier League 2023-24",
    }


def _build_match_semantic_content(row: dict) -> str:
    """Tạo câu mô tả tự nhiên cho một trận đấu và tỉ số cuối cùng."""
    home = _safe_field(row, "home_team")
    away = _safe_field(row, "away_team")
    score = _safe_field(row, "score")
    date = _safe_field(row, "date")
    venue = _safe_field(row, "venue")
    return f"On {date}, {home} played {away} at {venue} with a final score of {score}."


def _build_match_structured_content(row: dict) -> str:
    """Tạo chunk markdown chứa thông tin trận đấu theo từng field."""
    lines = ["# Match Details", "", f"- Date: {_safe_field(row, 'date')}", f"- Home team: {_safe_field(row, 'home_team')}", f"- Away team: {_safe_field(row, 'away_team')}", f"- Venue: {_safe_field(row, 'venue')}", f"- Score: {_safe_field(row, 'score')}"]
    return "\n".join(lines)


def _build_match_chunks(raw_dir: Path) -> list[dict]:
    """Tạo semantic + structured chunk cho từng trận."""
    df = pd.read_csv(raw_dir / "schedule.csv")
    chunks = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        doc_id = f"match_{idx}"
        metadata = _build_match_metadata(row_dict)
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_semantic",
            "chunk_type": "semantic",
            "doc_type": "match",
            "content": _build_match_semantic_content(row_dict),
            "metadata": metadata,
            "created_at": _now_ts(),
        })
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_structured",
            "chunk_type": "structured",
            "doc_type": "match",
            "content": _build_match_structured_content(row_dict),
            "metadata": metadata,
            "created_at": _now_ts(),
        })
    return chunks


def load_transfer_chunks(corpus_path: Path, transfers_dir: Path = None) -> list[dict]:
    """Đọc transfer chunks; nếu thiếu thì fallback sang CSV."""
    chunks = []
    if corpus_path.exists() and corpus_path.stat().st_size > 0:
        with open(corpus_path, encoding="utf-8") as fh:
            for line in fh:
                doc = json.loads(line)
                chunks.append({
                    "id":       doc["id"],
                    "text":     doc["text"],
                    "source":   "transfers",
                    "team":     None,
                    "metadata": {
                        "source_file":  doc.get("source_file", ""),
                        "chunk_index":  doc.get("chunk_index", 0),
                        "total_chunks": doc.get("total_chunks", 1),
                        "season":       doc.get("season", "2023-24"),
                    },
                })
        return chunks

    if transfers_dir is not None:
        chunks = (
            _load_transfer_chunks_from_csv(transfers_dir / "all_arrivals_2023_24.csv", "arrivals")
            + _load_transfer_chunks_from_csv(transfers_dir / "all_departures_2023_24.csv", "departures")
        )
    return chunks


def load_player_detail_chunks(raw_dir: Path, exclude_gk: bool = False) -> list[dict]:
    """Tạo chunk chi tiết từng dòng cho cầu thủ."""
    files = [
        raw_dir / "standard_stats.csv",
        raw_dir / "playing_time_stats.csv",
        raw_dir / "shooting_stats.csv",
        raw_dir / "misc_stats.csv",
    ]
    if exclude_gk:
        return [
            chunk
            for path in files
            for chunk in _load_csv_row_chunks(
                path,
                f"player_{path.stem}",
                lambda pos: str(pos).strip().upper() != "GK",
            )
        ]
    return [
        chunk
        for path in files
        for chunk in _load_csv_row_chunks(path, f"player_{path.stem}")
    ]


def load_keeper_detail_chunks(raw_dir: Path) -> list[dict]:
    """Tạo chunk chi tiết riêng cho thủ môn từ các bảng thống kê raw."""
    files = [
        raw_dir / "keeper_stats.csv",
        raw_dir / "misc_stats.csv",
        raw_dir / "playing_time_stats.csv",
        raw_dir / "shooting_stats.csv",
        raw_dir / "standard_stats.csv",
    ]
    return [
        chunk
        for path in files
        for chunk in _load_csv_row_chunks(
            path,
            f"keeper_{path.stem}",
            lambda pos: str(pos).strip().upper() == "GK",
        )
    ]
    

def load_player_stat_chunks(stats_path: Path) -> list[dict]:
    """Tạo chunk tổng hợp player stats theo từng đội."""
    df = pd.read_csv(stats_path)
    chunks = []

    for team, grp in df.groupby("team"):
        lines = [f"# {team} — Player Standard Stats (2023-24 Premier League)\n"]
        for _, row in grp.sort_values("Playing Time_Min", ascending=False).iterrows():
            lines.append(
                f"  - {row['player']} | Pos: {_fmt(row['pos'])} | Nat: {_fmt(row['nation'])} "
                f"| Age: {_fmt(row['age'])} | Apps: {_fmt(row['Playing Time_MP'])} "
                f"| Mins: {_fmt(row['Playing Time_Min'])} "
                f"| Goals: {_fmt(row['Performance_Gls'])} "
                f"| Assists: {_fmt(row['Performance_Ast'])} "
                f"| G+A: {_fmt(row['Performance_G+A'])} "
                f"| Yellow: {_fmt(row['Performance_CrdY'])} "
                f"| Red: {_fmt(row['Performance_CrdR'])}"
            )
        chunks.append({
            "id":       f"player_stats_{team.lower().replace(' ', '_')}",
            "text":     "\n".join(lines),
            "source":   "player_stats",
            "team":     team,
            "metadata": {"season": "2023-24", "num_players": len(grp)},
        })
    return chunks


def load_schedule_chunks(schedule_path: Path) -> list[dict]:
    """Tạo chunk lịch/kết quả theo từng đội."""
    df = pd.read_csv(schedule_path)
    # Keep only played matches (score is not NaN)
    played = df[df["score"].notna()].copy()

    team_names = pd.unique(played[["home_team", "away_team"]].values.ravel())
    chunks = []

    for team in sorted(team_names):
        home = played[played["home_team"] == team][["date", "home_team", "score", "away_team", "venue"]]
        away = played[played["away_team"] == team][["date", "home_team", "score", "away_team", "venue"]]
        all_games = pd.concat([home, away]).sort_values("date")

        lines = [f"# {team} — 2023-24 Premier League Results\n"]
        for _, row in all_games.iterrows():
            h, a = row["home_team"], row["away_team"]
            loc = "HOME" if h == team else "AWAY"
            lines.append(
                f"  [{row['date']}] {loc}: {h} {row['score']} {a}"
            )
        chunks.append({
            "id":       f"schedule_{team.lower().replace(' ', '_')}",
            "text":     "\n".join(lines),
            "source":   "schedule",
            "team":     team,
            "metadata": {"season": "2023-24", "num_games": len(all_games)},
        })
    return chunks


def _write_jsonl(path: Path, chunks: list[dict]) -> None:
    """Ghi danh sách chunk ra file JSONL, mỗi dòng là một JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write("\n")


def load_all_chunks(raw_dir: str | Path = None) -> list[dict]:
    """Load toàn bộ dữ liệu raw, tạo các nhóm chunk chính."""
    if raw_dir is None:
        raw_dir = Path(__file__).parent.parent / "data" / "raw"
    raw = Path(raw_dir)
    corpus_path = Path(__file__).parent.parent / "data" / "corpus.jsonl"
    transfer_raw_dir = raw / "epl_transfers"

    transfer_chunks = load_transfer_chunks(corpus_path, transfer_raw_dir)
    player_detail_chunks = load_player_detail_chunks(raw, exclude_gk=True)
    keeper_detail_chunks = load_keeper_detail_chunks(raw)
    player_doc_chunks = _build_player_chunks(raw)
    team_doc_chunks = _build_team_chunks(raw)
    team_meta_chunks = _build_club_metadata_chunks(raw)
    match_doc_chunks = _build_match_chunks(raw)
    player_stats_chunks = load_player_stat_chunks(raw / "standard_stats.csv")
    schedule_chunks = load_schedule_chunks(raw / "schedule.csv")

    all_chunks = (
        transfer_chunks
        + player_stats_chunks
        + schedule_chunks
        + player_detail_chunks
        + keeper_detail_chunks
    )

    print(f"[chunker] Loaded {len(all_chunks)} chunks total")
    by_source: dict[str, int] = {}
    for c in all_chunks:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    for src, count in sorted(by_source.items()):
        print(f"  {src:<20} {count:>4} chunks")

    output_dir = Path(__file__).parent.parent / "data" / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_dir / "transfers_chunks.jsonl", transfer_chunks)
    _write_jsonl(output_dir / "schedule_chunks.jsonl", schedule_chunks)
    _write_jsonl(output_dir / "all_players.jsonl", player_doc_chunks + player_stats_chunks + player_detail_chunks)
    _write_jsonl(output_dir / "all_teams.jsonl", team_doc_chunks + team_meta_chunks)
    _write_jsonl(output_dir / "all_matches.jsonl", match_doc_chunks)

    print(f"  Exported {len(transfer_chunks)} transfers chunks to {output_dir / 'transfers_chunks.jsonl'}")
    print(f"  Exported {len(schedule_chunks)} schedule chunks to {output_dir / 'schedule_chunks.jsonl'}")
    print(f"  Exported {len(player_doc_chunks) + len(player_stats_chunks) + len(player_detail_chunks)} all_players chunks to {output_dir / 'all_players.jsonl'}")
    print(f"  Exported {len(team_doc_chunks) + len(team_meta_chunks)} all_teams chunks to {output_dir / 'all_teams.jsonl'}")
    print(f"  Exported {len(match_doc_chunks)} all_matches chunks to {output_dir / 'all_matches.jsonl'}")

    return all_chunks


if __name__ == "__main__":
    chunks = load_all_chunks()
