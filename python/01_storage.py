import pandas as pd
import sqlite3
import os

CLEANED_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cleaned', 'cleaned_air_quality.csv')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cleaned', 'air_quality.db')
DB_PATH = os.path.abspath(DB_PATH)

df = pd.read_csv(CLEANED_CSV, parse_dates=['Date'])
print(f"Loaded {len(df)} rows from cleaned CSV")

conn = sqlite3.connect(DB_PATH)

df.to_sql('air_quality', conn, if_exists='replace', index=False)
print(f"Created table 'air_quality' with {len(df)} rows")

city_summary = df.groupby(['City', 'Country']).agg(
    avg_pm25=('PM2.5', 'mean'),
    avg_pm10=('PM10', 'mean'),
    avg_no2=('NO2', 'mean'),
    avg_so2=('SO2', 'mean'),
    avg_co=('CO', 'mean'),
    avg_o3=('O3', 'mean'),
    avg_temp=('Temperature', 'mean'),
    avg_humidity=('Humidity', 'mean'),
    avg_wind=('Wind Speed', 'mean'),
    num_records=('PM2.5', 'count')
).reset_index()

city_summary.to_sql('city_summary', conn, if_exists='replace', index=False)
print(f"Created table 'city_summary' with {len(city_summary)} rows")

print("\n--- Sample Queries ---")

print("\nTop 5 cities by average PM2.5:")
query1 = """
SELECT City, Country, ROUND(avg_pm25, 2) as avg_pm25, num_records
FROM city_summary
ORDER BY avg_pm25 DESC
LIMIT 5;
"""
result1 = pd.read_sql(query1, conn)
print(result1.to_string(index=False))

print("\nMonthly average PM2.5 (all cities):")
query2 = """
SELECT Month, ROUND(AVG("PM2.5"), 2) as avg_pm25, COUNT(*) as records
FROM air_quality
GROUP BY Month
ORDER BY Month;
"""
result2 = pd.read_sql(query2, conn)
print(result2.to_string(index=False))

print("\nCities exceeding WHO annual PM2.5 guideline (15 ug/m3):")
query3 = """
SELECT City, Country, ROUND(avg_pm25, 2) as avg_pm25
FROM city_summary
WHERE avg_pm25 > 15
ORDER BY avg_pm25 DESC;
"""
result3 = pd.read_sql(query3, conn)
print(result3.to_string(index=False))

conn.close()
print(f"\nDatabase saved to: {DB_PATH}")
