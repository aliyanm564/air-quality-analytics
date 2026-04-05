import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

CLEANED_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned', 'cleaned_air_quality.csv')
VISUALS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visuals')
os.makedirs(VISUALS_DIR, exist_ok=True)

df = pd.read_csv(CLEANED_CSV, parse_dates=['Date'])
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")

print("=" * 60)
print("1. SUMMARY STATISTICS")
print("=" * 60)
num_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
print(df[num_cols].describe().round(2))

print(f"\nMissing values per column:")
print(df.isnull().sum())
print(f"Total missing: {df.isnull().sum().sum()}")

print(f"\nAQI Category Distribution:")
print(df['AQI_Category'].value_counts().sort_index())

print("\n" + "=" * 60)
print("2. GENERATING DISTRIBUTION PLOTS")
print("=" * 60)

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle('Distribution of All Numeric Features', fontsize=14, fontweight='bold')
for idx, col in enumerate(num_cols):
    ax = axes[idx // 3][idx % 3]
    ax.hist(df[col], bins=40, color='#3498db', edgecolor='white', alpha=0.8)
    ax.set_title(col, fontsize=11)
    ax.set_ylabel('Frequency')
    ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1, label=f'Mean: {df[col].mean():.1f}')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: distributions.png")

fig, ax = plt.subplots(figsize=(10, 8))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title('Correlation Matrix of Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

print(f"\nKey correlations with PM2.5:")
pm25_corr = corr['PM2.5'].drop('PM2.5').sort_values(ascending=False)
for feat, val in pm25_corr.items():
    print(f"  {feat}: {val:.3f}")

print("\n" + "=" * 60)
print("4. CITY-LEVEL ANALYSIS")
print("=" * 60)

city_avg = df.groupby('City')[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].mean()
city_avg = city_avg.sort_values('PM2.5', ascending=False)
print("\nCities ranked by average PM2.5:")
print(city_avg['PM2.5'].round(2))

fig, ax = plt.subplots(figsize=(14, 6))
city_avg_sorted = city_avg.sort_values('PM2.5', ascending=True)
bars = ax.barh(city_avg_sorted.index, city_avg_sorted['PM2.5'], color='#e74c3c', alpha=0.8)
ax.axvline(x=15, color='green', linestyle='--', linewidth=2, label='WHO Annual Guideline (15 ug/m3)')
ax.set_xlabel('Average PM2.5 (ug/m3)')
ax.set_title('Average PM2.5 by City (2023)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'city_pm25_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: city_pm25_comparison.png")

monthly_avg = df.groupby('Month')[num_cols].mean()
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(monthly_avg.index, monthly_avg['PM2.5'], 'o-', color='#e74c3c', linewidth=2, label='PM2.5')
axes[0].plot(monthly_avg.index, monthly_avg['PM10'], 's-', color='#3498db', linewidth=2, label='PM10')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Concentration')
axes[0].set_title('Monthly Average Particulate Matter', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].set_xticks(range(1, 13))

axes[1].plot(monthly_avg.index, monthly_avg['Temperature'], 'o-', color='#e67e22', linewidth=2, label='Temperature (C)')
ax2 = axes[1].twinx()
ax2.plot(monthly_avg.index, monthly_avg['Humidity'], 's-', color='#2ecc71', linewidth=2, label='Humidity (%)')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Temperature (C)')
ax2.set_ylabel('Humidity (%)')
axes[1].set_title('Monthly Average Weather Conditions', fontsize=12, fontweight='bold')
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2)
axes[1].set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'seasonal_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: seasonal_patterns.png")

fig, ax = plt.subplots(figsize=(12, 6))
order = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy']
existing_cats = [c for c in order if c in df['AQI_Category'].values]
sns.boxplot(data=df, x='AQI_Category', y='Temperature', order=existing_cats, palette='RdYlGn_r', ax=ax)
ax.set_title('Temperature Distribution by AQI Category', fontsize=14, fontweight='bold')
ax.set_xlabel('AQI Category')
ax.set_ylabel('Temperature (C)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'aqi_temperature_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: aqi_temperature_boxplot.png")

print("\n" + "=" * 60)
print("KEY INSIGHT FOR PREDICTION")
print("=" * 60)
print("The correlation analysis shows that PM2.5 has very weak correlations")
print("with all other features (all |r| < 0.05). This suggests the dataset is")
print("synthetically generated with independent random columns. Despite this,")
print("we will still build a prediction model to demonstrate the methodology.")
print("A Random Forest may capture non-linear patterns a linear model misses.")
print("This insight will be discussed in the final report as a limitation.")
