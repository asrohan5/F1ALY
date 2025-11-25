import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12,6)
plt.rcParams['font.size']=10

def load_processed_data(race='Brazil', year=2024):
    base_path = 'D:/Projects/F1ALY/data/processed'

    race_pace = pd.read_parquet(f'{base_path}/{year}_{race}_race_pace.parquet')
    pit_data = pd.read_parquet(f'{base_path}/{year}_{race}_pit.parquet')
    driver_summary = pd.read_parquet(f'{base_path}/{year}_{race}_driver_summary.parquet')

    print(f'Loaded processed data for {year} {race}')
    print(f'Race pace laps: {len(race_pace)}')
    print(f'pit laps: {len(pit_data)}')
    print(f'Drivers: {len(driver_summary)}\n')

    return race_pace, pit_data, driver_summary
