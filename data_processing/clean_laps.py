import pandas as pd

def filter_laps(laps_df):
    required_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
    for col in required_cols:
        laps_df = laps_df[laps_df[col].notnull()]
    
    print(f"Filtered laps: {len(laps_df)} remaining")
    return laps_df

def extract_pitstop_laps(laps_df):
    pit_laps = laps_df[laps_df['PitOutTime'].notnull() | laps_df['PitInTime'].notnull()]
    print(f"Laps with pit stops: {len(pit_laps)}")
    return pit_laps

def summarize_driver_laps(laps_df):
    summary = laps_df.groupby('Driver')['LapTime'].count().reset_index()
    summary.columns = ['Driver', 'ValidLaps']
    print("\n Driver lap summary:\n", summary)
    return summary

def save_clean_data(laps_df, filename= 'clean_laps.parquet'):
    laps_df.to_parquet(filename)
    print(f"Saved cleaned laps to {filename}")


if __name__ == '__main__':
    from data_ingestion.fetch_data import load_session_from_disk
    laps_df, metadata = load_session_from_disk(2024, 'Brazil', 'R')

    clean_laps = filter_laps(laps_df)
    pit_laps = extract_pitstop_laps(laps_df)
    driver_summary = summarize_driver_laps(clean_laps)

    save_clean_data(clean_laps, 'D:/Projects/F1ALY/data/2024_Brazil_clean_laps.parquet')
    save_clean_data(pit_laps, 'D:/Projects/F1ALY/data/2024_Brazil_pit_laps.parquet')
    

    
