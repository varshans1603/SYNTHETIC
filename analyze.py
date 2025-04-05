import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log_file = 'prediction_log.csv'
df = pd.read_csv(log_file, engine='python')

# Fill missing if any
for col in ['Prediction', 'Timestamp']:
    if col not in df.columns:
        df[col] = None

# Convert Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Bar Chart - Prediction Summary
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='Prediction', palette='Set2')
plt.title('Fraud Prediction Summary')
plt.tight_layout()
plt.savefig('static/fraud_pie_chart.png')
plt.close()

print("✅ Analysis complete! Chart saved to 'static/fraud_pie_chart.png'")
