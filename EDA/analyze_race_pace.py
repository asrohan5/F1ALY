import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12,6)
plt.rcParams['font.size']=10

def load_processed_data(race='Brazil', year=2024):
    base_path = 'D:/Projects/F1ALY/data/processed'

    race_pace = pd.read_parquet(f'{base_path}/{year}_{race}_race.parquet')
    pit_data = pd.read_parquet(f'{base_path}/{year}_{race}_pit.parquet')
    driver_summary = pd.read_parquet(f'{base_path}/{year}_{race}_driver_summary.parquet')

    print(f'Loaded processed data for {year} {race}')
    print(f'Race pace laps: {len(race_pace)}')
    print(f'pit laps: {len(pit_data)}')
    print(f'Drivers: {len(driver_summary)}\n')

    return race_pace, pit_data, driver_summary


#Which Driver had the most consistent race pace?
def analyze_driver_consistency(driver_summary):
    print("Driver Consistency Analysis")
    consistent = driver_summary.sort_values('StdLap').head(10)
    print('\n Top 10 Most Consistent Drivers (lowest lap time variation): ')
    print(consistent[['Driver', 'ValidLaps', 'AvgLap', 'StdLap', 'BestLap']])


    print("\n Business Insight: ")
    print("Drivers with low StdLap delivered predictable pace throughout the race")
    print('This indicates good tire management and consistent car setup\n')

    return consistent

def visualize_pace_distribution(race_pace_df):

    top_drivers = (race_pace_df.groupby('Driver')['LapTimeSeconds'].count().sort_values(ascending=False).head(10).index.tolist())
    data = race_pace_df[race_pace_df['Driver'].isin(top_drivers)]

    plt.figure(figsize=(14,8))
    sns.boxplot(data=data, x='Driver', y='LapTimeSeconds', palette='Set2')
    plt.title('Race Pace Distribution - Top 10 Drivers by Lap Count', fontsize=14, fontweight='bold')
    plt.xlabel('Driver', fontsize=12)
    plt.ylabel('Lap Time (seconds)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs('D:/Projects/F1ALY/EDA/plots', exist_ok = True)
    plt.savefig('D:/Projects/F1ALY/EDA/plots/pace_distribution.png', dpi=300)
    print('Saved pace_distribution.png')
    plt.close()

def visualize_lap_progression(race_pace_df):
    top_5 = (
        race_pace_df.groupby('Driver')['LapTimeSeconds'].count().sort_values(ascending=False).head(5).index.tolist()
    )

    plt.figure(figsize=(14,8))

    for driver in top_5:
        driver_data = race_pace_df[race_pace_df['Driver'] == driver].sort_values('LapNumber')
        plt.plot(driver_data['LapNumber'], driver_data['LapTimeSeconds'], marker = 'o', label=driver, alpha=0.7)
        plt.title('Lap Time Progression - Top 5 drivers', fontsize = 14, fontweight='bold')
        plt.xlabel('LapNumber', fontsize=12)
        plt.ylabel('Lap Time (seconds)', fontsize = 12)
        plt.legend(title = 'Driver')
        plt.grid(True, alpha = 0.3)
        plt.tight_layout()

        plt.savefig(f'D:/Projects/F1ALY/EDA/plots/lap_progression_{driver}.png', dpi=300)
        print('Saved lap_progression.png')
        plt.close()

#How many pit stops did each driver make?
def analyze_pit_strategy(pit_data, driver_summary):
    print('Pit Stop Strategy Analysis')

    pit_counts = pit_data.groupby('Driver').size().reset_index(name='PitStops')
    pit_counts = pit_counts.sort_values('PitStops', ascending=False)

    print("\n Pit Stops by Driver: ")
    print(pit_counts.head(15))

    print('\n Business Insight: ')
    print('Drivers with more pit stops may have chosen aggressive tire strategies')
    print('or experienced tire issues requiring extra stops\n')

    return pit_counts

if __name__ == '__main__':
    race_pace_df, pit_df, driver_summary_df = load_processed_data('Brazil', 2024)

    print('Starting EDA :::')

    consistent_drivers = analyze_driver_consistency(driver_summary_df)

    pit_summary = analyze_pit_strategy(pit_df, driver_summary_df)

    visualize_pace_distribution(race_pace_df)
    visualize_lap_progression(race_pace_df)

    print('EDA Completed:::')




