import pandas as pd
import numpy as np
import os

def clean_data():
    raw_file = "iot_sensor_raw_data_extended.csv"
    cleaned_file = "cleaned_iot_sensor_data.csv"
    
    if not os.path.exists(raw_file):
        print(f"Error: {raw_file} not found.")
        return
        
    print("==================================================")
    print("Starting Data Cleaning Process for IoT Sensor Data")
    print("==================================================")
    
    # 1. Load Raw Data
    df_raw = pd.read_csv(raw_file)
    initial_rows = len(df_raw)
    print(f"1. Loaded raw data: {initial_rows} rows.")
    
    # 2. Check for missing device_id (ห้ามเป็นค่าว่าง)
    # We define missing device_id as NaN, empty string, or 'NULL'
    missing_dev_mask = df_raw['device_id'].isna() | (df_raw['device_id'].astype(str).str.strip().str.upper() == 'NULL') | (df_raw['device_id'].astype(str).str.strip() == '')
    missing_dev_count = missing_dev_mask.sum()
    print(f"2. Found {missing_dev_count} rows with missing device_id (to be removed).")
    
    # 3. Detect duplicates (device_id + timestamp เดียวกัน)
    # Count duplicates in the raw dataset
    raw_duplicates_count = df_raw.duplicated(subset=['device_id', 'timestamp'], keep='first').sum()
    print(f"3. Found {raw_duplicates_count} duplicate rows in raw data (to be removed).")
    
    # Start cleaning process
    df_clean = df_raw[~missing_dev_mask].copy()
    
    # Drop duplicates
    rows_before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['device_id', 'timestamp'], keep='first')
    dedup_removed = rows_before_dedup - len(df_clean)
    print(f"   - Removed {dedup_removed} duplicate rows after dropping missing device_id.")
    
    # 4. Outliers detection (แปลงค่าผิดปกติเป็น NULL/NaN)
    # temp_c outlier (not between 0 and 50)
    df_clean['temp_c'] = pd.to_numeric(df_clean['temp_c'], errors='coerce')
    temp_outliers_mask = (df_clean['temp_c'] < 0) | (df_clean['temp_c'] > 50)
    temp_outliers_count = temp_outliers_mask.sum()
    df_clean.loc[temp_outliers_mask, 'temp_c'] = np.nan
    print(f"4. Found {temp_outliers_count} temp_c outliers (converted to NULL).")
    
    # battery outlier (not between 0 and 100)
    df_clean['battery'] = pd.to_numeric(df_clean['battery'], errors='coerce')
    battery_outliers_mask = (df_clean['battery'] < 0) | (df_clean['battery'] > 100)
    battery_outliers_count = battery_outliers_mask.sum()
    df_clean.loc[battery_outliers_mask, 'battery'] = np.nan
    print(f"   - Found {battery_outliers_count} battery outliers (converted to NULL).")
    
    # 5. Normalize motion and handle invalid formats
    # valid formats: true, truee, yes -> True; false, falsee, 0 -> False
    # others (unknown, NULL, etc.) -> NaN
    def parse_motion(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s in ['true', 'truee', 'yes']:
            return True
        elif s in ['false', 'falsee', '0']:
            return False
        else:
            return np.nan
            
    motion_before_normalization = df_clean['motion'].copy()
    df_clean['motion'] = df_clean['motion'].apply(parse_motion)
    
    # Count invalid motion formats that were converted to NaN (excluding already NaN)
    motion_invalid_count = ((motion_before_normalization.notna()) & (df_clean['motion'].isna())).sum()
    print(f"   - Found {motion_invalid_count} invalid motion formats (converted to NULL).")
    
    # 6. Count missing values before imputation
    temp_missing_before = df_clean['temp_c'].isna().sum()
    battery_missing_before = df_clean['battery'].isna().sum()
    motion_missing_before = df_clean['motion'].isna().sum()
    print(f"5. Missing values count (before imputation):")
    print(f"   - temp_c: {temp_missing_before}")
    print(f"   - battery: {battery_missing_before}")
    print(f"   - motion: {motion_missing_before}")
    
    # Track which rows are being cleaned/imputed to set status
    # Any row with any originally missing value or converted outlier is marked as 'Cleaned'
    # To do this, we compare the current df_clean before imputation with final
    # Let's save a mask of rows with missing values
    needs_imputation_mask = df_clean['temp_c'].isna() | df_clean['battery'].isna() | df_clean['motion'].isna()
    
    # 7. Impute Missing Values
    # Sort values chronologically per device to ensure correct forward fill
    df_clean = df_clean.sort_values(by=['device_id', 'timestamp']).reset_index(drop=True)
    needs_imputation_mask = df_clean['temp_c'].isna() | df_clean['battery'].isna() | df_clean['motion'].isna()
    
    # Imputation: forward fill, then backward fill for each device group
    df_clean['temp_c'] = df_clean.groupby('device_id')['temp_c'].ffill().bfill()
    df_clean['battery'] = df_clean.groupby('device_id')['battery'].ffill().bfill()
    df_clean['motion'] = df_clean.groupby('device_id')['motion'].ffill().bfill()
    
    # 8. Add data_quality_status column
    df_clean['data_quality_status'] = np.where(needs_imputation_mask, 'Cleaned', 'Good')
    
    final_rows = len(df_clean)
    print("==================================================")
    print("Cleaning Summary Report")
    print("==================================================")
    print(f"Initial Rows: {initial_rows}")
    print(f"Final Cleaned Rows: {final_rows}")
    print(f"Rows Deleted: {initial_rows - final_rows} (Missing device_id: {missing_dev_count}, Duplicates: {dedup_removed})")
    print("\nAnomalies & Quality Issues Solved:")
    print(f"- Missing Device IDs (Dropped): {missing_dev_count}")
    print(f"- Duplicates Removed: {dedup_removed}")
    print(f"- Temperature Outliers (Converted & Imputed): {temp_outliers_count}")
    print(f"- Battery Outliers (Converted & Imputed): {battery_outliers_count}")
    print(f"- Missing/Invalid Motion Values (Normalized & Imputed): {motion_missing_before}")
    print(f"- Missing Temp/Battery (Imputed): Temp: {temp_missing_before - temp_outliers_count}, Battery: {battery_missing_before - battery_outliers_count}")
    
    print("\nData Quality Status Distribution in Cleaned Data:")
    print(df_clean['data_quality_status'].value_counts())
    
    # Save to CSV
    df_clean.to_csv(cleaned_file, index=False)
    print(f"\nCleaned data successfully saved to: {cleaned_file}")
    print("==================================================")

if __name__ == "__main__":
    clean_data()
