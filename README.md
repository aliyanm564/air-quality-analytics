# Global Air Quality Analytics Pipeline

An end-to-end data analytics pipeline analyzing global air quality data from 20 major cities worldwide.

## Analytics Question
**Which cities have the worst air quality, and what factors drive those differences?**

Secondary: Can we predict PM2.5 concentrations from weather and other pollutant data?

## Tools Used
| Tool | Purpose |
|------|---------|
| Microsoft Excel | ETL / Data Cleaning / Data Dictionary |
| Python (pandas, scikit-learn, matplotlib) | EDA, Prediction Model, SQLite Storage |
| Tableau | Interactive Dashboard / Visualizations |

## Dataset
- **Source:** [Kaggle - Global Air Quality Data](https://www.kaggle.com/datasets)
- **Size:** 10,000 rows × 12 columns
- **License:** Open Data (CC0 / Public Domain)
- **Features:** PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, Wind Speed across 20 cities

## Project Structure
```
air-quality-analytics/
├── README.md
├── proposal/
│   └── proposal.pdf
├── data/
│   ├── raw/
│   │   └── global_air_quality_data_10000.csv
│   └── cleaned/
│       ├── cleaned_air_quality.csv
│       └── air_quality.db
├── excel/
│   └── data_cleaning.xlsx
├── python/
│   ├── requirements.txt
│   ├── 01_storage.py
│   ├── 02_eda.py
│   └── 03_prediction.py
├── tableau/
│   ├── Global Air Quality (2023).twbx
├── report/
│   └── final_report.pdf
└── visuals/
    ├── distributions.png
    ├── correlation_heatmap.png
    ├── city_pm25_comparison.png
    ├── seasonal_patterns.png
    ├── aqi_temperature_boxplot.png
    ├── actual_vs_predicted.png
    ├── feature_importance.png
    ├── residuals.png
    ├── model_comparison.png
    ├── Tableau Dashboard.png
    └── model_results.json
```

## How to Reproduce

### Prerequisites
- Python 3.8+
- pip
- Tableau Desktop or Tableau Public (free)

### Step 1: Clone and Setup
```bash
git clone https://github.com/YOUR_USERNAME/air-quality-analytics.git
cd air-quality-analytics
pip install -r python/requirements.txt
```

### Step 2: Data Cleaning (Excel)
Open `excel/data_cleaning.xlsx` to review the ETL process:
- **Sheet 1 (Raw Data):** Sample of original data
- **Sheet 2 (Cleaning Log):** All cleaning decisions documented
- **Sheet 3 (Cleaned Data):** Sample of cleaned data with derived columns
- **Sheet 4 (Data Dictionary):** Field descriptions and transformations
- **Sheet 5 (Summary Stats):** Descriptive statistics

The cleaned CSV is already provided at `data/cleaned/cleaned_air_quality.csv`.

### Step 3: Storage (Python + SQLite)
```bash
python python/01_storage.py
```
Creates `data/cleaned/air_quality.db` with two tables and demonstrates SQL queries.

### Step 4: EDA (Python)
```bash
python python/02_eda.py
```
Generates distribution plots, correlation heatmap, city comparisons, and seasonal pattern charts in `visuals/`.

### Step 5: Prediction (Python + scikit-learn)
```bash
python python/03_prediction.py
```
Trains Linear Regression (baseline) and Random Forest (improved) models. Outputs evaluation metrics and visualizations.

### Step 6: Dashboard (Tableau)
Follow the instructions in `tableau/tableau_instructions.md` to build the interactive dashboard. Connect to `data/cleaned/cleaned_air_quality.csv`.

## Key Findings
1. All 20 cities exceed the WHO annual PM2.5 guideline of 15 ug/m3
2. Dubai (80.01), Sydney (78.93), and Mumbai (78.90) have the highest average PM2.5
3. Features show near-zero correlations, indicating synthetic data generation
4. Prediction models achieve negative R² values due to lack of feature-target relationships
5. The pipeline methodology is sound and would produce meaningful results with real-world data

## Limitations
- Dataset is synthetically generated (uniform distributions, no inter-variable correlations)
- Only 20 cities represented
- Some temperature values unrealistic for certain cities
- Full methodology discussion in the final report
