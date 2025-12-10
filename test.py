'''
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

features_path = 'D:/Projects./F1ALY/modeling/2024_Brazil_features.parquet'
df = pd.read_parquet(features_path)
print(df.columns)

'''
'''
X = df[['TireAge', 'LapTimeDelta', 'ConsistencyScore', 'Compound_Encoded', 'RaceProgress%']].fillna(0)
y = df['LapTimeSeconds']


'''

import fastf1
schedule = fastf1.get_event_schedule(2024)
print(schedule[['RoundNumber', 'EventName']])
