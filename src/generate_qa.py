from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict


CHUNKS_DIR = "../data/chunks"
QA_DIR = "../data/qa"
SEED = 42
TEST_RATIO = 0.30

qa_pairs: list[dict] = []


def load_jsonl(path: str) -> list[dict]:
    """Đọc file JSONL và trả về danh sách dict, bỏ qua dòng rỗng."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def add_qa(question: str, answer: str, qtype: str, difficulty: str = "easy", source_chunk_id: str = "") -> None:
    """Thêm một QA pair hợp lệ vào biến toàn cục qa_pairs."""
    if not question or answer is None:
        return
    answer = str(answer).strip()
    if answer in ("nan", "None", "", "0.0", "N/A"):
        return
    qa_pairs.append({
        "question": question.strip(),
        "answer": answer,
        "type": qtype,
        "difficulty": difficulty,
        "source_chunk_id": source_chunk_id,
    })


def generate_from_transfers() -> None:
    """Sinh câu hỏi từ transfer chunks"""
    records = load_jsonl(f"{CHUNKS_DIR}/transfers_chunks.jsonl")

    for rec in records:
        text = rec.get("text", "")
        cid = rec.get("id", "")

        player_m = re.search(r"(?:Arrivals|Departures)\s*[—–]\s*(.+)", text)
        team_m = re.search(r"Team:\s*([^\|]+)", text)
        fee_m = re.search(r"Fee:\s*([^\n\|]+)", text)
        from_m = re.search(r"From:\s*([^\n]+)", text)
        to_m = re.search(r"To:\s*([^\n]+)", text)
        pos_m = re.search(r"Position:\s*([^\n\|]+)", text)
        nat_m = re.search(r"Nation:\s*([^\n\|]+)", text)
        age_m = re.search(r"Age:\s*(\d+)", text)
        league_m = re.search(r"League:\s*([^\n\|]+)", text)

        player = player_m.group(1).strip() if player_m else ""
        team = team_m.group(1).strip() if team_m else rec.get("team", "")
        fee = fee_m.group(1).strip() if fee_m else ""
        from_ = from_m.group(1).strip() if from_m else ""
        to_ = to_m.group(1).strip() if to_m else ""
        pos = pos_m.group(1).strip() if pos_m else ""
        nat = nat_m.group(1).strip() if nat_m else ""
        age = age_m.group(1).strip() if age_m else ""
        league = league_m.group(1).strip() if league_m else ""

        if not player or not team:
            continue

        is_arrival = "Arrivals" in text

        if fee and "£" in fee:
            add_qa(f"What was the transfer fee for {player}?", fee, "transfer_fee", "easy", cid)
            if is_arrival:
                add_qa(f"How much did {team} pay for {player}?", fee, "transfer_fee", "easy", cid)
            else:
                add_qa(f"How much did {team} receive for {player}?", fee, "transfer_fee", "easy", cid)

        if from_ and is_arrival:
            add_qa(f"Which club did {player} join {team} from?", from_, "transfer_origin", "easy", cid)
            add_qa(f"Where did {player} transfer from before joining {team}?", from_, "transfer_origin", "easy", cid)

        if to_ and not is_arrival:
            add_qa(f"Where did {player} move after leaving {team}?", to_, "transfer_dest", "easy", cid)
            add_qa(f"Which club did {player} join after {team}?", to_, "transfer_dest", "easy", cid)

        if pos:
            add_qa(f"What position does {player} play?", pos, "player_position", "easy", cid)
        if nat:
            add_qa(f"What is {player}'s nationality?", nat, "player_nationality", "easy", cid)
        if age:
            add_qa(f"How old was {player} when he {'joined' if is_arrival else 'left'} {team}?", age, "player_age", "easy", cid)
        if league and is_arrival and from_:
            add_qa(f"From which league did {team} sign {player}?", league, "transfer_league", "medium", cid)


def generate_from_players() -> None:
    """Sinh câu hỏi về thống kê của cầu thủ"""
    records = load_jsonl(f"{CHUNKS_DIR}/all_players.jsonl")

    for rec in records:
        text = rec.get("content", "")
        cid = rec.get("chunk_id") or rec.get("doc_id", "")
        meta = rec.get("metadata", {})

        player = meta.get("player_name", "")
        team = meta.get("team", "")
        pos = meta.get("position", "")
        nat = meta.get("nationality", "")

        if not player or not team:
            continue

        goals_m = re.search(r"scored (\d+) goals?", text)
        assists_m = re.search(r"provided (\d+) assists?", text)
        apps_m = re.search(r"made (\d+) appearances?", text)
        mins_m = re.search(r"playing (\d+(?:,\d+)?) minutes?", text)

        goals = goals_m.group(1) if goals_m else ""
        assists = assists_m.group(1) if assists_m else ""
        apps = apps_m.group(1) if apps_m else ""
        mins = mins_m.group(1).replace(",", "") if mins_m else ""

        if goals and int(goals) > 0:
            add_qa(f"How many goals did {player} score in the 2023-24 Premier League season?", goals, "player_goals", "easy", cid)
            add_qa(f"How many goals did {player} score for {team} in 2023-24?", goals, "player_goals", "easy", cid)
        if assists and int(assists) > 0:
            add_qa(f"How many assists did {player} provide in 2023-24?", assists, "player_assists", "easy", cid)
        if apps:
            add_qa(f"How many appearances did {player} make in the 2023-24 season?", apps, "player_apps", "easy", cid)
        if mins and int(mins) > 500:
            add_qa(f"How many minutes did {player} play in 2023-24?", mins, "player_minutes", "medium", cid)
        if pos:
            add_qa(f"What position does {player} play at {team}?", pos, "player_position", "easy", cid)
        if nat:
            add_qa(f"What is {player}'s nationality?", nat, "player_nationality", "easy", cid)
        add_qa(f"Which club did {player} play for in the 2023-24 Premier League?", team, "player_club", "easy", cid)


def generate_from_matches() -> None:
    """Sinh câu hỏi về trận đấu."""
    records = load_jsonl(f"{CHUNKS_DIR}/all_matches.jsonl")

    for rec in records:
        cid = rec.get("chunk_id") or rec.get("doc_id", "")
        meta = rec.get("metadata", {})

        home = meta.get("home_team", "")
        away = meta.get("away_team", "")
        score = meta.get("score", "")
        venue = meta.get("venue", "")
        date = meta.get("date", "")
        if not home or not away or not score:
            continue

        winner = ""
        score_parts = parse_score(score)
        if score_parts:
            hg, ag = score_parts
            if hg > ag:
                winner = home
            elif ag > hg:
                winner = away
            else:
                winner = "Draw"

        add_qa(f"What was the score when {home} played {away} on {date}?", score, "match_score", "easy", cid)
        if venue:
            add_qa(f"Where did {home} play {away}?", venue, "match_venue", "easy", cid)
        if winner and winner != "Draw":
            add_qa(f"Who won the match between {home} and {away} on {date}?", winner, "match_winner", "easy", cid)
            add_qa(f"Did {home} beat {away} on {date}?", "Yes" if winner == home else "No", "match_result_yn", "easy", cid)
        if winner == "Draw":
            add_qa(f"What was the result of the match between {home} and {away} on {date}?", "Draw", "match_result_type", "easy", cid)


def generate_from_schedule() -> None:
    """Sinh câu hỏi từ schedule chunks."""
    records = load_jsonl(f"{CHUNKS_DIR}/schedule_chunks.jsonl")

    for rec in records:
        text = rec.get("text", "")
        cid = rec.get("id", "")
        team = rec.get("team", "")
        if not team:
            continue

        wins = len(re.findall(r"\[W\]", text))
        draws = len(re.findall(r"\[D\]", text))
        losses = len(re.findall(r"\[L\]", text))
        total = wins + draws + losses

        if total > 0:
            add_qa(f"How many games did {team} win in the 2023-24 Premier League?", str(wins), "team_wins", "medium", cid)
            add_qa(f"How many games did {team} lose in the 2023-24 Premier League?", str(losses), "team_losses", "medium", cid)
            add_qa(f"How many draws did {team} have in the 2023-24 season?", str(draws), "team_draws", "medium", cid)
            add_qa(f"What was {team}'s win-draw-loss record in 2023-24?", f"{wins}W-{draws}D-{losses}L", "team_record", "medium", cid)


def generate_from_teams() -> None:
    """Sinh câu hỏi về thống kê của đội."""
    records = load_jsonl(f"{CHUNKS_DIR}/all_teams.jsonl")

    for rec in records:
        cid = rec.get("chunk_id") or rec.get("doc_id", "")
        meta = rec.get("metadata", {})

        team = meta.get("team_name", "")
        goals = meta.get("goals", "")
        assists = meta.get("assists", "")
        poss = meta.get("possession", "")
        ycards = meta.get("yellow_cards", "")

        if not team:
            continue
        if goals:
            add_qa(f"How many goals did {team} score in the 2023-24 Premier League?", goals, "team_goals", "easy", cid)
        if assists:
            add_qa(f"How many assists did {team} create in 2023-24?", assists, "team_assists", "medium", cid)
        if poss:
            add_qa(f"What was {team}'s possession percentage in 2023-24?", f"{poss}%", "team_possession", "medium", cid)
        if ycards:
            add_qa(f"How many yellow cards did {team} receive in 2023-24?", ycards, "team_yellow_cards", "medium", cid)


def generate_comparative() -> None:
    """Sinh câu hỏi hard tổng hợp."""
    records = load_jsonl(f"{CHUNKS_DIR}/all_players.jsonl")
    teams_records = load_jsonl(f"{CHUNKS_DIR}/all_teams.jsonl")

    player_goals: dict[str, int] = {}
    for rec in records:
        meta = rec.get("metadata", {})
        text = rec.get("content", "")
        player = meta.get("player_name", "")
        gm = re.search(r"scored (\d+) goals?", text)
        if player and gm:
            player_goals[player] = max(player_goals.get(player, 0), int(gm.group(1)))

    if player_goals:
        top_scorer = max(player_goals, key=player_goals.get)
        add_qa("Who was the top scorer in the 2023-24 Premier League?", top_scorer, "comparative_goals", "hard")
        add_qa("How many goals did the top scorer score in the 2023-24 Premier League?", str(player_goals[top_scorer]), "comparative_goals", "hard")

    team_goals: dict[str, int] = {}
    team_poss: dict[str, float] = {}
    for rec in teams_records:
        meta = rec.get("metadata", {})
        team = meta.get("team_name", "")
        if not team:
            continue
        try:
            team_goals[team] = int(float(meta.get("goals", "")))
        except (TypeError, ValueError):
            pass
        try:
            team_poss[team] = float(meta.get("possession", ""))
        except (TypeError, ValueError):
            pass

    if team_goals:
        add_qa("Which team scored the most goals in the 2023-24 Premier League?", max(team_goals, key=team_goals.get), "comparative_team_goals", "hard")
    if team_poss:
        add_qa("Which team had the highest possession in the 2023-24 Premier League?", max(team_poss, key=team_poss.get), "comparative_possession", "hard")


def normalize_entity_text(text: str) -> str:
    """Chuẩn hóa text thành khóa entity cho split train/test để tránh leakage."""
    text = text.lower().strip()
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    return re.sub(r"_+", "_", text).strip("_")


def strip_chunk_suffix(chunk_id: str) -> str:
    """Bỏ hậu tố khỏi chunk id."""
    return re.sub(r"_(semantic|structured|metadata)$", "", chunk_id)


def entity_id_for_pair(qa: dict) -> str:
    """Gán mỗi QA vào một entity để tránh cùng entity xuất hiện ở cả train và test."""
    if qa.get("source_entity_id"):
        return qa["source_entity_id"]

    cid = qa.get("source_chunk_id", "")
    qtype = qa.get("type", "unknown")
    question = qa.get("question", "")

    if not cid:
        return f"global:{qtype}"

    cid = strip_chunk_suffix(cid)
    if cid.startswith("match_"):
        match = re.match(r"(match_\d+)", cid)
        return match.group(1) if match else cid
    if cid.startswith("player_") or cid.startswith("team_") or cid.startswith("schedule_"):
        return cid
    if cid.startswith("transfer_"):
        match = re.match(r"transfer_(arrivals|departures)_(.+)_\d+$", cid)
        if match:
            direction, player_key = match.groups()
            return f"transfer_{direction}_{player_key}"
        return cid
    return f"{qtype}:{normalize_entity_text(question)}"


def safe_int(value) -> int | None:
    """Ép giá trị sang int."""
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def safe_float(value) -> float | None:
    """Ép giá trị sang float."""
    try:
        return float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def parse_score(score: str) -> tuple[int, int] | None:
    """Parse tỉ số thành tuple"""
    parts = re.split(r"[–-]", str(score))
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def make_hard_qa(question: str, answer: str, qtype: str, entity_id: str, source_chunk_id: str = "") -> dict:
    """Tạo dict QA hard."""
    return {
        "question": question.strip(),
        "answer": str(answer).strip(),
        "type": qtype,
        "difficulty": "hard",
        "source_chunk_id": source_chunk_id,
        "source_entity_id": entity_id,
    }


def load_unique_players() -> list[dict]:
    """Đọc player semantic chunks và rút ra mỗi cầu thủ một record thống kê."""
    players: dict[str, dict] = {}
    for rec in load_jsonl(f"{CHUNKS_DIR}/all_players.jsonl"):
        if rec.get("chunk_type") != "semantic":
            continue

        text = rec.get("content", "")
        meta = rec.get("metadata", {})
        player = meta.get("player_name", "")
        team = meta.get("team", "")
        if not player or not team:
            continue

        goals_m = re.search(r"scored (\d+) goals?", text)
        assists_m = re.search(r"provided (\d+) assists?", text)
        apps_m = re.search(r"made (\d+) appearances?", text)
        mins_m = re.search(r"playing (\d+(?:,\d+)?) minutes?", text)
        chunk_id = rec.get("chunk_id") or rec.get("doc_id", "")

        players[player] = {
            "player": player,
            "team": team,
            "goals": safe_int(goals_m.group(1)) if goals_m else None,
            "assists": safe_int(assists_m.group(1)) if assists_m else None,
            "apps": safe_int(apps_m.group(1)) if apps_m else None,
            "minutes": safe_int(mins_m.group(1)) if mins_m else None,
            "chunk_id": chunk_id,
            "entity_id": strip_chunk_suffix(chunk_id),
        }
    return list(players.values())


def load_unique_teams() -> list[dict]:
    """Đọc team semantic chunks và rút ra mỗi đội một record thống kê."""
    teams: dict[str, dict] = {}
    for rec in load_jsonl(f"{CHUNKS_DIR}/all_teams.jsonl"):
        if rec.get("chunk_type") != "semantic":
            continue

        meta = rec.get("metadata", {})
        team = meta.get("team_name", "")
        if not team:
            continue

        chunk_id = rec.get("chunk_id") or rec.get("doc_id", "")
        teams[team] = {
            "team": team,
            "goals": safe_int(meta.get("goals")),
            "assists": safe_int(meta.get("assists")),
            "possession": safe_float(meta.get("possession")),
            "yellow_cards": safe_int(meta.get("yellow_cards")),
            "chunk_id": chunk_id,
            "entity_id": strip_chunk_suffix(chunk_id),
        }
    return list(teams.values())


def load_unique_matches() -> list[dict]:
    """Đọc match semantic chunks và rút ra mỗi trận một record có score đã parse."""
    matches: dict[str, dict] = {}
    for rec in load_jsonl(f"{CHUNKS_DIR}/all_matches.jsonl"):
        if rec.get("chunk_type") != "semantic":
            continue

        meta = rec.get("metadata", {})
        home = meta.get("home_team", "")
        away = meta.get("away_team", "")
        score = meta.get("score", "")
        score_parts = parse_score(score)
        if not home or not away or not score_parts:
            continue

        chunk_id = rec.get("chunk_id") or rec.get("doc_id", "")
        home_goals, away_goals = score_parts
        matches[strip_chunk_suffix(chunk_id)] = {
            "home": home,
            "away": away,
            "score": score,
            "date": meta.get("date", ""),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "chunk_id": chunk_id,
            "entity_id": strip_chunk_suffix(chunk_id),
        }
    return list(matches.values())


def comparison_answer(left_name: str, left_value, right_name: str, right_value) -> str | None:
    """Trả về tên entity có giá trị lớn hơn."""
    if left_value is None or right_value is None or left_value == right_value:
        return None
    return left_name if left_value > right_value else right_name


def generate_hard_team_comparisons() -> list[dict]:
    """Sinh câu hard so sánh 2 đội theo các chỉ số thống kê."""
    teams = sorted(load_unique_teams(), key=lambda item: item["team"])
    metrics = [
        ("goals", "scored more goals", "comparative_team_goals"),
        ("assists", "created more assists", "comparative_team_assists"),
        ("possession", "had higher possession", "comparative_team_possession"),
        ("yellow_cards", "received more yellow cards", "comparative_team_yellow_cards"),
    ]

    hard: list[dict] = []
    for idx, left in enumerate(teams):
        for right in teams[idx + 1:]:
            entity_id = f"comparison_team:{left['entity_id']}:{right['entity_id']}"
            for metric, phrase, qtype in metrics:
                answer = comparison_answer(left["team"], left[metric], right["team"], right[metric])
                if answer:
                    hard.append(make_hard_qa(
                        f"Between {left['team']} and {right['team']}, which team {phrase} in the 2023-24 Premier League?",
                        answer,
                        qtype,
                        entity_id,
                        left["chunk_id"],
                    ))
    return hard


def generate_hard_player_comparisons(limit_per_metric: int = 260) -> list[dict]:
    """Sinh câu hard so sánh cầu thủ theo các chỉ số thống kê."""
    players = load_unique_players()
    rng = random.Random(SEED)
    metrics = [
        ("goals", "scored more goals", "comparative_player_goals"),
        ("assists", "provided more assists", "comparative_player_assists"),
        ("minutes", "played more minutes", "comparative_player_minutes"),
        ("apps", "made more appearances", "comparative_player_apps"),
    ]

    hard: list[dict] = []
    for metric, phrase, qtype in metrics:
        candidates: list[tuple[dict, dict]] = []
        for idx, left in enumerate(players):
            for right in players[idx + 1:]:
                if left[metric] is not None and right[metric] is not None and left[metric] != right[metric]:
                    candidates.append((left, right))
        rng.shuffle(candidates)

        for left, right in candidates[:limit_per_metric]:
            answer = comparison_answer(left["player"], left[metric], right["player"], right[metric])
            if answer:
                entity_id = f"comparison_player:{left['entity_id']}:{right['entity_id']}"
                hard.append(make_hard_qa(
                    f"Between {left['player']} and {right['player']}, who {phrase} in the 2023-24 Premier League?",
                    answer,
                    qtype,
                    entity_id,
                    left["chunk_id"],
                ))
    return hard


def generate_hard_match_arithmetic() -> list[dict]:
    """Sinh câu hard cần tính toán từ tỉ số trận đấu."""
    hard: list[dict] = []
    for match in load_unique_matches():
        home_goals = match["home_goals"]
        away_goals = match["away_goals"]
        total_goals = home_goals + away_goals
        margin = abs(home_goals - away_goals)
        if home_goals > away_goals:
            outcome = f"{match['home']} won by {margin} goals"
        elif away_goals > home_goals:
            outcome = f"{match['away']} won by {margin} goals"
        else:
            outcome = "Draw"

        hard.append(make_hard_qa(
            f"How many total goals were scored when {match['home']} played {match['away']} on {match['date']}?",
            str(total_goals),
            "match_total_goals",
            match["entity_id"],
            match["chunk_id"],
        ))
        hard.append(make_hard_qa(
            f"What was the goal difference in the match between {match['home']} and {match['away']} on {match['date']}?",
            str(margin),
            "match_goal_difference",
            match["entity_id"],
            match["chunk_id"],
        ))
        hard.append(make_hard_qa(
            f"Based on the score {match['score']}, what was the outcome of {match['home']} vs {match['away']} on {match['date']}?",
            outcome,
            "match_outcome_from_score",
            match["entity_id"],
            match["chunk_id"],
        ))
    return hard


def generate_additional_hard_questions() -> list[dict]:
    """Gộp toàn bộ nhóm hard questions bổ sung ngoài các template cơ bản."""
    hard = []
    hard.extend(generate_hard_team_comparisons())
    hard.extend(generate_hard_player_comparisons())
    hard.extend(generate_hard_match_arithmetic())
    return hard


def dedup(pairs: list[dict]) -> list[dict]:
    """Loại câu hỏi trùng nhau dựa trên nội dung question đã lowercase."""
    seen, unique = set(), []
    for qa in pairs:
        key = qa["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(qa)
    return unique


def build_all_pairs() -> list[dict]:
    """Chạy tất cả generator, deduplicate và gán source_entity_id cho từng QA."""
    qa_pairs.clear()
    random.seed(SEED)

    print("\nGenerating Q&A from each source:\n")
    generate_from_transfers()
    generate_from_players()
    generate_from_matches()
    generate_from_schedule()
    generate_from_teams()
    generate_comparative()

    pairs = dedup(qa_pairs + generate_additional_hard_questions())
    for qa in pairs:
        qa["source_entity_id"] = entity_id_for_pair(qa)
    return pairs


def primary_type(items: list[dict]) -> str:
    """Lấy question type phổ biến nhất trong một nhóm entity."""
    counts = Counter(item.get("type", "unknown") for item in items)
    return counts.most_common(1)[0][0]


def entity_split(pairs: list[dict], test_ratio: float = TEST_RATIO) -> tuple[list[dict], list[dict], dict]:
    """Chia train/test theo entity để không leakage giữa hai split."""
    rng = random.Random(SEED)
    groups: dict[str, list[dict]] = defaultdict(list)
    for qa in pairs:
        groups[qa["source_entity_id"]].append(qa)

    groups_by_type: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for entity, items in groups.items():
        groups_by_type[primary_type(items)].append((entity, items))

    train_entities, test_entities = set(), set()
    for _, typed_groups in sorted(groups_by_type.items()):
        rng.shuffle(typed_groups)
        total_questions = sum(len(items) for _, items in typed_groups)
        target_test = round(total_questions * test_ratio)
        current_test = 0
        max_test_groups = len(typed_groups) - 1 if len(typed_groups) > 1 else len(typed_groups)

        for idx, (entity, items) in enumerate(typed_groups):
            if idx < max_test_groups and current_test < target_test:
                test_entities.add(entity)
                current_test += len(items)
            else:
                train_entities.add(entity)

    overlap = train_entities & test_entities
    if overlap:
        raise AssertionError(f"Entity leakage detected: {sorted(overlap)[:10]}")

    train_qa = [qa for qa in pairs if qa["source_entity_id"] in train_entities]
    test_qa = [qa for qa in pairs if qa["source_entity_id"] in test_entities]
    rng.shuffle(train_qa)
    rng.shuffle(test_qa)

    metadata = {
        "seed": SEED,
        "test_ratio": test_ratio,
        "total_pairs": len(pairs),
        "train_pairs": len(train_qa),
        "test_pairs": len(test_qa),
        "total_entities": len(groups),
        "train_entities": len(train_entities),
        "test_entities": len(test_entities),
        "entity_overlap_count": len(overlap),
        "train_entity_ids": sorted(train_entities),
        "test_entity_ids": sorted(test_entities),
    }
    return train_qa, test_qa, metadata


def save_split(pairs: list[dict], split_dir: str, name: str) -> None:
    """Ghi một split ra questions.txt, reference_answers.txt và qa_pairs.json."""
    os.makedirs(split_dir, exist_ok=True)
    with open(f"{split_dir}/questions.txt", "w", encoding="utf-8") as qf, \
         open(f"{split_dir}/reference_answers.txt", "w", encoding="utf-8") as af:
        for qa in pairs:
            qf.write(qa["question"] + "\n")
            af.write(qa["answer"] + "\n")
    with open(f"{split_dir}/qa_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"  {name}: {len(pairs)} pairs -> {split_dir}/")


def print_distribution(train_qa: list[dict], test_qa: list[dict], all_pairs: list[dict]) -> None:
    """In thống kê số lượng câu hỏi theo từng type trong train/test."""
    print("\nType distribution:")
    all_types = sorted({qa["type"] for qa in all_pairs})
    for qtype in all_types:
        total = sum(1 for qa in all_pairs if qa["type"] == qtype)
        train = sum(1 for qa in train_qa if qa["type"] == qtype)
        test = sum(1 for qa in test_qa if qa["type"] == qtype)
        print(f"  {qtype:<28} total={total:>5}  train={train:>4}  test={test:>4}")


def save_iaa_subset(test_qa: list[dict]) -> None:
    """Lấy ngẫu nhiên tối đa 20 câu test để hai annotator dùng cho IAA."""
    rng = random.Random(SEED)
    iaa = rng.sample(test_qa, min(20, len(test_qa)))
    iaa_path = f"{QA_DIR}/iaa_subset.json"
    os.makedirs(QA_DIR, exist_ok=True)
    with open(iaa_path, "w", encoding="utf-8") as f:
        json.dump({
            "instructions": (
                "Two annotators answer each question independently. "
                "This subset comes from the entity-level test split."
            ),
            "questions": [
                {
                    "id": i + 1,
                    "question": qa["question"],
                    "type": qa["type"],
                    "source_entity_id": qa["source_entity_id"],
                }
                for i, qa in enumerate(iaa)
            ],
        }, f, ensure_ascii=False, indent=2)


def main() -> None:
    pairs = build_all_pairs()
    train_qa, test_qa, metadata = entity_split(pairs)

    print(f"\nTotal unique Q&A pairs: {len(pairs)}")
    print_distribution(train_qa, test_qa, pairs)

    save_split(train_qa, f"{QA_DIR}/train", "TRAIN")
    save_split(test_qa, f"{QA_DIR}/test", "TEST")
    save_iaa_subset(test_qa)

    with open(f"{QA_DIR}/split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
