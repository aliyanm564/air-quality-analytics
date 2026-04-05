import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os
import json

CLEANED_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned', 'cleaned_air_quality.csv')
VISUALS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visuals')

df = pd.read_csv(CLEANED_CSV, parse_dates=['Date'])
print(f"Dataset: {df.shape[0]} rows")

le = LabelEncoder()
df['City_Encoded'] = le.fit_transform(df['City'])

feature_cols = ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed', 'Month', 'City_Encoded']
target = 'PM2.5'

X = df[feature_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")

print("\n" + "=" * 60)
print("BASELINE MODEL: Linear Regression")
print("=" * 60)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"MAE:  {mae_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"R2:   {r2_lr:.4f}")

print("\n" + "=" * 60)
print("IMPROVED MODEL: Random Forest Regressor")
print("=" * 60)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"MAE:  {mae_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R2:   {r2_rf:.4f}")

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"{'Metric':<10} {'Linear Regression':>20} {'Random Forest':>20}")
print("-" * 52)
print(f"{'MAE':<10} {mae_lr:>20.2f} {mae_rf:>20.2f}")
print(f"{'RMSE':<10} {rmse_lr:>20.2f} {rmse_rf:>20.2f}")
print(f"{'R-squared':<10} {r2_lr:>20.4f} {r2_rf:>20.4f}")

improvement = ((mae_lr - mae_rf) / mae_lr) * 100
print(f"\nRandom Forest MAE improvement over baseline: {improvement:.1f}%")

results = {
    "linear_regression": {"MAE": round(mae_lr, 2), "RMSE": round(rmse_lr, 2), "R2": round(r2_lr, 4)},
    "random_forest": {"MAE": round(mae_rf, 2), "RMSE": round(rmse_rf, 2), "R2": round(r2_rf, 4)},
    "improvement_pct": round(improvement, 1)
}
with open(os.path.join(VISUALS_DIR, 'model_results.json'), 'w') as f:
    json.dump(results, f, indent=2)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, preds, name in [(axes[0], y_pred_lr, 'Linear Regression'), (axes[1], y_pred_rf, 'Random Forest')]:
    ax.scatter(y_test, preds, alpha=0.3, s=10, color='#3498db')
    ax.plot([0, 150], [0, 150], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual PM2.5')
    ax.set_ylabel('Predicted PM2.5')
    ax.set_title(f'{name}\nMAE={mean_absolute_error(y_test, preds):.2f}, R2={r2_score(y_test, preds):.4f}')
    ax.legend()
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 160)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: actual_vs_predicted.png")

importances = rf.feature_importances_
feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='#2ecc71', alpha=0.8)
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance for PM2.5 Prediction', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

fig, ax = plt.subplots(figsize=(10, 5))
residuals = y_test - y_pred_rf
ax.hist(residuals, bins=50, color='#9b59b6', alpha=0.8, edgecolor='white')
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Residual (Actual - Predicted)')
ax.set_ylabel('Frequency')
ax.set_title('Random Forest Residual Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'residuals.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: residuals.png")

fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['MAE', 'RMSE']
lr_vals = [mae_lr, rmse_lr]
rf_vals = [mae_rf, rmse_rf]
x = np.arange(len(metrics))
width = 0.3
ax.bar(x - width/2, lr_vals, width, label='Linear Regression', color='#3498db', alpha=0.8)
ax.bar(x + width/2, rf_vals, width, label='Random Forest', color='#2ecc71', alpha=0.8)
ax.set_ylabel('Error Value')
ax.set_title('Model Comparison: Error Metrics', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
for i, (lr_v, rf_v) in enumerate(zip(lr_vals, rf_vals)):
    ax.text(i - width/2, lr_v + 0.3, f'{lr_v:.1f}', ha='center', fontsize=9)
    ax.text(i + width/2, rf_v + 0.3, f'{rf_v:.1f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("Both models show low R-squared values, confirming the EDA insight")
print("that features in this synthetic dataset are largely independent.")
print("The Random Forest slightly outperforms the Linear Regression,")
print("which is expected as it can capture non-linear relationships.")
print("In a real-world dataset, we would expect much stronger correlations")
print("between weather/pollutant features and PM2.5 concentrations.")
