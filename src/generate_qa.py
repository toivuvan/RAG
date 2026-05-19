"""
STEP 3 — Q&A ANNOTATION GENERATION
=====================================
Tạo bộ câu hỏi - đáp án từ corpus, đảm bảo:
  • Đa dạng loại câu hỏi (6 loại)
  • Phân bổ đều theo source_type
  • Không trùng lặp câu hỏi
  • Split train/test 70/30 stratified

Output:
  data/qa/train/questions.txt
  data/qa/train/reference_answers.txt
  data/qa/train/qa_pairs.json
  data/qa/test/questions.txt
  data/qa/test/reference_answers.txt
  data/qa/test/qa_pairs.json
  data/qa/iaa_subset.json

Chạy: python src/03_generate_qa.py
"""

import json, os, re, random
from collections import defaultdict
from pathlib import Path

CHUNKS_DIR = "../data/chunks"
QA_DIR     = "../data/qa"
os.makedirs(f"{QA_DIR}/train", exist_ok=True)
os.makedirs(f"{QA_DIR}/test",  exist_ok=True)

random.seed(42)

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

qa_pairs = []

def add_qa(question: str, answer: str, qtype: str,
           difficulty: str = "easy", source_chunk_id: str = ""):
    if not question or not answer:
        return
    answer = str(answer).strip()
    if answer in ("nan", "None", "", "0.0", "N/A"):
        return
    qa_pairs.append({
        "question":        question.strip(),
        "answer":          answer,
        "type":            qtype,
        "difficulty":      difficulty,
        "source_chunk_id": source_chunk_id,
    })


# 1. TRANSFERS (transfers_chunks.jsonl)

def generate_from_transfers():
    records = load_jsonl(f"{CHUNKS_DIR}/transfers_chunks.jsonl")
    print(f"  Transfers: {len(records)} chunks")

    for rec in records:
        text = rec.get("text", "")
        cid  = rec.get("id", "")
        meta = rec.get("metadata", {})

        # Parse structured fields from text
        player_m = re.search(r"(?:Arrivals|Departures)\s*[—–]\s*(.+)", text)
        team_m   = re.search(r"Team:\s*([^\|]+)", text)
        fee_m    = re.search(r"Fee:\s*([^\n\|]+)", text)
        from_m   = re.search(r"From:\s*([^\n]+)", text)
        to_m     = re.search(r"To:\s*([^\n]+)", text)
        pos_m    = re.search(r"Position:\s*([^\n\|]+)", text)
        nat_m    = re.search(r"Nation:\s*([^\n\|]+)", text)
        age_m    = re.search(r"Age:\s*(\d+)", text)
        league_m = re.search(r"League:\s*([^\n\|]+)", text)

        player = player_m.group(1).strip() if player_m else ""
        team   = (team_m.group(1).strip() if team_m else rec.get("team",""))
        fee    = fee_m.group(1).strip()    if fee_m    else ""
        from_  = from_m.group(1).strip()   if from_m   else ""
        to_    = to_m.group(1).strip()     if to_m     else ""
        pos    = pos_m.group(1).strip()    if pos_m    else ""
        nat    = nat_m.group(1).strip()    if nat_m    else ""
        age    = age_m.group(1).strip()    if age_m    else ""
        league = league_m.group(1).strip() if league_m else ""

        if not player or not team:
            continue

        is_arrival = "Arrivals" in text

        # Q1: Transfer fee
        if fee and "£" in fee:
            add_qa(f"What was the transfer fee for {player}?", fee, "transfer_fee", "easy", cid)
            if is_arrival:
                add_qa(f"How much did {team} pay for {player}?", fee, "transfer_fee", "easy", cid)
            else:
                add_qa(f"How much did {team} receive for {player}?", fee, "transfer_fee", "easy", cid)

        # Q2: Origin club
        if from_ and is_arrival:
            add_qa(f"Which club did {player} join {team} from?", from_, "transfer_origin", "easy", cid)
            add_qa(f"Where did {player} transfer from before joining {team}?", from_, "transfer_origin", "easy", cid)

        # Q3: Destination
        if to_ and not is_arrival:
            add_qa(f"Where did {player} move after leaving {team}?", to_, "transfer_dest", "easy", cid)
            add_qa(f"Which club did {player} join after {team}?", to_, "transfer_dest", "easy", cid)

        # Q4: Position
        if pos:
            add_qa(f"What position does {player} play?", pos, "player_position", "easy", cid)

        # Q5: Nationality
        if nat:
            add_qa(f"What is {player}'s nationality?", nat, "player_nationality", "easy", cid)

        # Q6: Age
        if age:
            add_qa(f"How old was {player} when he {'joined' if is_arrival else 'left'} {team}?",
                   age, "player_age", "easy", cid)

        # Q7: League origin
        if league and is_arrival and from_:
            add_qa(f"From which league did {team} sign {player}?", league, "transfer_league", "medium", cid)


# 2. PLAYER STATS (all_players.jsonl)

def generate_from_players():
    records = load_jsonl(f"{CHUNKS_DIR}/all_players.jsonl")
    print(f"  Players: {len(records)} chunks")

    for rec in records:
        text = rec.get("content", "")
        cid  = rec.get("chunk_id") or rec.get("doc_id", "")
        meta = rec.get("metadata", {})

        player = meta.get("player_name", "")
        team   = meta.get("team", "")
        pos    = meta.get("position", "")
        age    = str(meta.get("age", ""))
        nat    = meta.get("nationality", "")

        if not (player and team):
            continue

        # Parse stats from text
        goals_m   = re.search(r"scored (\d+) goals?", text)
        assists_m = re.search(r"provided (\d+) assists?", text)
        apps_m    = re.search(r"made (\d+) appearances?", text)
        mins_m    = re.search(r"playing (\d+(?:,\d+)?) minutes?", text)
        tackles_m = re.search(r"made (\d+) tackles?", text)

        goals   = goals_m.group(1)   if goals_m   else ""
        assists = assists_m.group(1) if assists_m else ""
        apps    = apps_m.group(1)    if apps_m    else ""
        mins    = mins_m.group(1).replace(",","") if mins_m else ""

        # Q1: Goals
        if goals and int(goals) > 0:
            add_qa(f"How many goals did {player} score in the 2023-24 Premier League season?",
                   goals, "player_goals", "easy", cid)
            add_qa(f"How many goals did {player} score for {team} in 2023-24?",
                   goals, "player_goals", "easy", cid)

        # Q2: Assists
        if assists and int(assists) > 0:
            add_qa(f"How many assists did {player} provide in 2023-24?",
                   assists, "player_assists", "easy", cid)

        # Q3: Appearances
        if apps:
            add_qa(f"How many appearances did {player} make in the 2023-24 season?",
                   apps, "player_apps", "easy", cid)

        # Q4: Minutes played
        if mins and int(mins) > 500:
            add_qa(f"How many minutes did {player} play in 2023-24?",
                   mins, "player_minutes", "medium", cid)

        # Q5: Position
        if pos:
            add_qa(f"What position does {player} play at {team}?", pos, "player_position", "easy", cid)

        # Q6: Nationality
        if nat:
            add_qa(f"What is {player}'s nationality?", nat, "player_nationality", "easy", cid)

        # Q7: Club
        add_qa(f"Which club did {player} play for in the 2023-24 Premier League?",
               team, "player_club", "easy", cid)


# 3. MATCH RESULTS (all_matches.jsonl)

def generate_from_matches():
    records = load_jsonl(f"{CHUNKS_DIR}/all_matches.jsonl")
    print(f"  Matches: {len(records)} chunks")

    for rec in records:
        text = rec.get("content", "")
        cid  = rec.get("chunk_id") or rec.get("doc_id", "")
        meta = rec.get("metadata", {})

        home  = meta.get("home_team", "")
        away  = meta.get("away_team", "")
        score = meta.get("score", "")
        venue = meta.get("venue", "")
        date  = meta.get("date", "")
        if not (home and away and score):
            continue

        # Parse result
        winner = ""
        if "–" in score:
            parts = score.split("–")
            try:
                hg, ag = int(parts[0]), int(parts[1])
                if hg > ag: winner = home
                elif ag > hg: winner = away
                else: winner = "Draw"
            except: pass

        # Q1: Score
        add_qa(f"What was the score when {home} played {away} on {date}?",
               score, "match_score", "easy", cid)

        # Q2: Venue
        if venue:
            add_qa(f"Where did {home} play {away}?",
                   venue, "match_venue", "easy", cid)

        # Q3: Winner
        if winner and winner != "Draw":
            add_qa(f"Who won the match between {home} and {away} on {date}?",
                   winner, "match_winner", "easy", cid)
            add_qa(f"Did {home} beat {away} on {date}?",
                   "Yes" if winner == home else "No", "match_result_yn", "easy", cid)

        # Q4: Draw
        if winner == "Draw":
            add_qa(f"What was the result of the match between {home} and {away} on {date}?",
                   "Draw", "match_result_type", "easy", cid)


# 4. SCHEDULE (schedule_chunks.jsonl)

def generate_from_schedule():
    records = load_jsonl(f"{CHUNKS_DIR}/schedule_chunks.jsonl")
    print(f"  Schedule: {len(records)} chunks")

    for rec in records:
        text = rec.get("text", "")
        cid  = rec.get("id", "")
        team = rec.get("team", "")
        if not team:
            continue

        # Count W/D/L from text
        wins   = len(re.findall(r"\[W\]", text))
        draws  = len(re.findall(r"\[D\]", text))
        losses = len(re.findall(r"\[L\]", text))
        total  = wins + draws + losses

        if total > 0:
            add_qa(f"How many games did {team} win in the 2023-24 Premier League?",
                   str(wins), "team_wins", "medium", cid)
            add_qa(f"How many games did {team} lose in the 2023-24 Premier League?",
                   str(losses), "team_losses", "medium", cid)
            add_qa(f"How many draws did {team} have in the 2023-24 season?",
                   str(draws), "team_draws", "medium", cid)
            add_qa(f"What was {team}'s win-draw-loss record in 2023-24?",
                   f"{wins}W-{draws}D-{losses}L", "team_record", "medium", cid)


# 5. TEAM STATS (all_teams.jsonl)

def generate_from_teams():
    records = load_jsonl(f"{CHUNKS_DIR}/all_teams.jsonl")
    print(f"  Teams: {len(records)} chunks")

    for rec in records:
        text = rec.get("content", "")
        cid  = rec.get("chunk_id") or rec.get("doc_id", "")
        meta = rec.get("metadata", {})

        team    = meta.get("team_name", "")
        goals   = meta.get("goals", "")
        assists = meta.get("assists", "")
        poss    = meta.get("possession", "")
        ycards  = meta.get("yellow_cards", "")
        rcards  = meta.get("red_cards", "")

        if not team:
            continue

        if goals:
            add_qa(f"How many goals did {team} score in the 2023-24 Premier League?",
                   goals, "team_goals", "easy", cid)

        if assists:
            add_qa(f"How many assists did {team} create in 2023-24?",
                   assists, "team_assists", "medium", cid)

        if poss:
            add_qa(f"What was {team}'s possession percentage in 2023-24?",
                   f"{poss}%", "team_possession", "medium", cid)

        if ycards:
            add_qa(f"How many yellow cards did {team} receive in 2023-24?",
                   ycards, "team_yellow_cards", "medium", cid)


# 6. COMPARATIVE / AGGREGATE QUESTIONS (hard)

def generate_comparative():
    # Load all players for aggregation
    records = load_jsonl(f"{CHUNKS_DIR}/all_players.jsonl")
    teams_records = load_jsonl(f"{CHUNKS_DIR}/all_teams.jsonl")

    # Top scorer
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
        top_goals  = player_goals[top_scorer]
        add_qa("Who was the top scorer in the 2023-24 Premier League?",
               top_scorer, "comparative_goals", "hard")
        add_qa(f"How many goals did the top scorer score in the 2023-24 Premier League?",
               str(top_goals), "comparative_goals", "hard")

    # Team with most goals
    team_goals: dict[str, int] = {}
    for rec in teams_records:
        meta = rec.get("metadata", {})
        team  = meta.get("team_name", "")
        goals = meta.get("goals", "")
        if team and goals:
            try: team_goals[team] = int(float(goals))
            except: pass

    if team_goals:
        top_team = max(team_goals, key=team_goals.get)
        add_qa("Which team scored the most goals in the 2023-24 Premier League?",
               top_team, "comparative_team_goals", "hard")

    # Team with highest possession
    team_poss: dict[str, float] = {}
    for rec in teams_records:
        meta = rec.get("metadata", {})
        team = meta.get("team_name", "")
        poss = meta.get("possession", "")
        if team and poss:
            try: team_poss[team] = float(poss)
            except: pass

    if team_poss:
        top_poss_team = max(team_poss, key=team_poss.get)
        add_qa("Which team had the highest possession in the 2023-24 Premier League?",
               top_poss_team, "comparative_possession", "hard")

    print(f"  Comparative: {len([q for q in qa_pairs if q['difficulty']=='hard'])} hard questions added")

#  SPLIT & SAVE

def dedup(pairs: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for qa in pairs:
        key = qa["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(qa)
    return unique


def save_split(pairs: list[dict], split_dir: str, name: str):
    os.makedirs(split_dir, exist_ok=True)
    with open(f"{split_dir}/questions.txt", "w", encoding="utf-8") as qf, \
         open(f"{split_dir}/reference_answers.txt", "w", encoding="utf-8") as af:
        for qa in pairs:
            qf.write(qa["question"] + "\n")
            af.write(qa["answer"]   + "\n")
    with open(f"{split_dir}/qa_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"  {name}: {len(pairs)} pairs → {split_dir}/")

#  MAIN

def main():
    print("=" * 58)
    print("  STEP 3: Q&A ANNOTATION GENERATION")
    print("=" * 58)
    print("\nGenerating Q&A from each source:\n")

    generate_from_transfers()
    generate_from_players()
    generate_from_matches()
    generate_from_schedule()
    generate_from_teams()
    generate_comparative()

    # Dedup
    unique = dedup(qa_pairs)
    random.shuffle(unique)
    print(f"\nTotal unique Q&A pairs: {len(unique)}")

    # Stratified split 70/30 by question type
    by_type = defaultdict(list)
    for qa in unique:
        by_type[qa["type"]].append(qa)

    train_qa, test_qa = [], []
    for qtype, items in by_type.items():
        random.shuffle(items)
        split = max(1, int(len(items) * 0.7))
        train_qa.extend(items[:split])
        test_qa.extend(items[split:])

    random.shuffle(train_qa)
    random.shuffle(test_qa)

    print(f"Train: {len(train_qa)} | Test: {len(test_qa)}")
    print("\nType distribution:")
    for qtype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        n_train = sum(1 for q in train_qa if q["type"] == qtype)
        n_test  = sum(1 for q in test_qa  if q["type"] == qtype)
        print(f"  {qtype:<28} total={len(items):>5}  train={n_train:>4}  test={n_test:>4}")

    save_split(train_qa, f"{QA_DIR}/train", "TRAIN")
    save_split(test_qa,  f"{QA_DIR}/test",  "TEST")

    # IAA subset: 20 câu từ test set
    iaa = random.sample(test_qa, min(20, len(test_qa)))
    iaa_path = f"{QA_DIR}/iaa_subset.json"
    with open(iaa_path, "w", encoding="utf-8") as f:
        json.dump({
            "instructions": (
                "Two annotators answer each question independently. "
                "Answers are compared to compute Cohen's Kappa."
            ),
            "questions": [
                {"id": i+1, "question": qa["question"], "type": qa["type"]}
                for i, qa in enumerate(iaa)
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"\nIAA subset (20 Q) saved: {iaa_path}")

    print(f"""
{'='*58}
  ANNOTATION COMPLETE ✓
  Train: {len(train_qa):,} pairs
  Test : {len(test_qa):,} pairs
  IAA  : 20 questions
{'='*58}
→ Next: python src/04_rag_pipeline.py
""")


if __name__ == "__main__":
    main()
