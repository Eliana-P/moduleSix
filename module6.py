import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('Traffic.csv')

print(df.head())
print(df.info())

print(df['Day of the week'].unique())


day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

df['day_of_week'] = df['Day of the week'].map(day_mapping)

df['date_time'] = pd.to_datetime(df['Day of the week'] + ' ' + df['Time'], format='%A %I:%M:%S %p')
df['date'] = df['date_time'].dt.date  
df['hour'] = df['date_time'].dt.hour  
df['month'] = df['date_time'].dt.month  

df['hour_label'] = df['hour'].apply(lambda x: 'Midnight' if x == 0 else f'{x}:00')

# 4. Filter data for past dates (up to May 10th)
df = df[df['date'] <= pd.to_datetime('2025-05-10').date()]

target = 'Total'  
features = ['hour', 'day_of_week', 'month']

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

df['predicted_volume'] = model.predict(df[features])


days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for day in range(7):
    daily_data = df[df['day_of_week'] == day]
    
    if daily_data.empty:
        print(f"No data for {days_of_week[day]}")
    else:
        plt.figure(figsize=(10, 6))

        sns.lineplot(x='hour', y='predicted_volume', data=daily_data, marker='o')

        plt.title(f'Predicted Traffic Volume by Hour for {days_of_week[day]}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Predicted Traffic Volume')
        plt.grid(True)
        plt.xticks(range(0, 24)) 
        plt.show()
