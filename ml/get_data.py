
import pandas as pd
import logging
import os


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/data_cleaning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def clean_usgs_data(file_path):
    print("Loading and auditing USGS earthquake data...")
    df = pd.read_csv(file_path)
    
    initial_count = len(df)
    logging.info(f"Starting audit on {initial_count} rows from USGS.")


    required_columns = ['magnitude', 'depth', 'latitude', 'longitude', 'time']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Critical column missing from dataset: {col}")

   
    valid_mask = (
        df['magnitude'].notnull() & (df['magnitude'] >= 0) &
        df['depth'].notnull() & (df['depth'] >= 0) &
        df['latitude'].notnull() & (df['longitude'].notnull())
    )


    df_valid = df[valid_mask].copy()
    df_rejected = df[~valid_mask].copy()

    if not df_rejected.empty:
        logging.warning(f"Rejected {len(df_rejected)} rows due to invalid values or missing fields.")
     
        df_rejected.to_csv("logs/rejected_earthquake_rows.csv", index=False)
    else:

    df_valid['time'] = pd.to_datetime(df_valid['time'])
    

    df_valid['Year'] = df_valid['time'].dt.year
    df_valid['Month'] = df_valid['time'].dt.month

    print(f"Audit complete. Valid rows: {len(df_valid)}, Logged & Rejected rows: {len(df_rejected)}")
    
    os.makedirs("data/processed", exist_ok=True)
    df_valid.to_csv("data/processed/cleaned_quakes.csv", index=False)
    return df_valid

if __name__ == "__main__":
    clean_usgs_data("data/usgs_sf_quakes.csv")
