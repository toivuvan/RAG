import soccerdata as sd
import pandas as pd
import os

# Đảm bảo thư mục lưu trữ tồn tại
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def flatten_columns(df):
    """Flatten multi-level columns from FBref"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
    return df

def fetch_player_stats():
    try:
        print("Đang kết nối tới FBref (mùa giải 23-24)...")
        # Khởi tạo API của soccerdata
        fbref = sd.FBref(leagues="ENG-Premier League", seasons="2324")
        
        # 1. Lấy bảng Standard Stats (Bàn thắng, kiến tạo, số trận...)
        print("Đang tải bảng Thống kê Cơ bản (Standard)...")
        df_standard = fbref.read_player_season_stats(stat_type="standard")
        df_standard = flatten_columns(df_standard)
        df_standard.reset_index(inplace=True)
        df_standard.to_csv(f"{OUTPUT_DIR}/standard_stats.csv", index=False)
        
        # 2. Lấy bảng Shooting (Số cú sút, xG...)
        print("Đang tải bảng Sút bóng (Shooting)...")
        df_shooting = fbref.read_player_season_stats(stat_type="shooting")
        df_shooting = flatten_columns(df_shooting)
        df_shooting.reset_index(inplace=True)
        df_shooting.to_csv(f"{OUTPUT_DIR}/shooting_stats.csv", index=False)
        
        # 3. Lấy bảng Playing Time (Thời gian chơi...)
        print("Đang tải bảng Thời gian chơi (Playing Time)...")
        df_playing_time = fbref.read_player_season_stats(stat_type="playing_time")
        df_playing_time = flatten_columns(df_playing_time)
        df_playing_time.reset_index(inplace=True)
        df_playing_time.to_csv(f"{OUTPUT_DIR}/playing_time_stats.csv", index=False)
        
        # 4. Lấy bảng Thủ môn (Keeper)
        print("Đang tải bảng Thủ môn (Keeper)...")
        df_keeper = fbref.read_player_season_stats(stat_type="keeper")
        df_keeper = flatten_columns(df_keeper)
        df_keeper.reset_index(inplace=True)
        df_keeper.to_csv(f"{OUTPUT_DIR}/keeper_stats.csv", index=False)
        
        # 5. Lấy bảng Thống kê linh tinh (Misc)
        print("Đang tải bảng Thống kê linh tinh (Misc)...")
        df_misc = fbref.read_player_season_stats(stat_type="misc")
        df_misc = flatten_columns(df_misc)
        df_misc.reset_index(inplace=True)
        df_misc.to_csv(f"{OUTPUT_DIR}/misc_stats.csv", index=False)
        
        # 6. Lấy bảng Thống kê Đội bóng (Team Stats)
        print("Đang tải bảng Thống kê Đội bóng (Team Stats)...")
        df_team = fbref.read_team_season_stats(stat_type="standard")
        df_team = flatten_columns(df_team)
        df_team.reset_index(inplace=True)
        df_team.to_csv(f"{OUTPUT_DIR}/team_stats.csv", index=False)
        
        # 7. Lấy bảng Sút bóng Đội (Team Shooting)
        print("Đang tải bảng Sút bóng Đội (Team Shooting)...")
        df_team_shooting = fbref.read_team_season_stats(stat_type="shooting")
        df_team_shooting = flatten_columns(df_team_shooting)
        df_team_shooting.reset_index(inplace=True)
        df_team_shooting.to_csv(f"{OUTPUT_DIR}/team_shooting_stats.csv", index=False)
        
        # 8. Lấy Lịch thi đấu & Kết quả (Schedule)
        print("Đang tải Lịch thi đấu & Kết quả...")
        df_schedule = fbref.read_schedule()
        df_schedule = flatten_columns(df_schedule)
        df_schedule.reset_index(inplace=True)
        df_schedule.to_csv(f"{OUTPUT_DIR}/schedule.csv", index=False)
        
        # 9. Lấy bảng Thời gian chơi Đội (Team Playing Time)
        print("Đang tải bảng Thời gian chơi Đội (Team Playing Time)...")
        df_team_playing_time = fbref.read_team_season_stats(stat_type="playing_time")
        df_team_playing_time = flatten_columns(df_team_playing_time)
        df_team_playing_time.reset_index(inplace=True)
        df_team_playing_time.to_csv(f"{OUTPUT_DIR}/team_playing_time_stats.csv", index=False)
        
        # 10. Lấy thống kê Đội Đằc biệt (Team Misc)
        print("Đang tải thống kê Đội Đằc biệt...")
        df_team_misc = fbref.read_team_season_stats(stat_type="misc")
        df_team_misc = flatten_columns(df_team_misc)
        df_team_misc.reset_index(inplace=True)
        df_team_misc.to_csv(f"{OUTPUT_DIR}/team_misc_stats.csv", index=False)
        
        print(f"Hoàn tất! Dữ liệu đã được lưu tại thư mục: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_player_stats()