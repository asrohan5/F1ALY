import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

features_path = 'D:/Projects./F1ALY/modeling/2024_Brazil_features.parquet'
df = pd.read_parquet(features_path)

X = df[['TireAge', 'LapTimeDelta', 'ConsistencyScore', 'Compound_Encoded', 'RaceProgress%']].fillna(0)
y = df['LapTimeSeconds']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Test MAE: {mae:.3f} and R2: {r2:.3f}')

fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print('n\Feature Importances:\n')
print(fi)

os.makedirs('D:/Projects/F1ALY/modeling/plots1', exist_ok = True)
plt.figure(figsize=(8,4))
sns.barplot(x = fi.values, y=fi.index, palette='mako')
plt.title('Feature Importance: Lap Time PRediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('D:/Projects/F1ALY/modeling/plots1/fi.png', dpi=200)
print('Saved fi.png:::::')

