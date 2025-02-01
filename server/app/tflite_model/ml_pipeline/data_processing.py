# /ml_pipeline/data_processing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(csv_file):
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Convert DateTime to datetime object
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Extract time-related features
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Drop duplicates and missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    # Normalize the 'Vehicles' column for ML model
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Vehicles_Normalized'] = scaler.fit_transform(df[['Vehicles']])
    
    return df, scaler

if __name__ == "__main__":
    # Specify input CSV file
    csv_file = "traffic_data.csv"
    
    # Process the data
    cleaned_data, scaler = load_and_clean_data(csv_file)
    
    # Save cleaned data to a new CSV file
    cleaned_data.to_csv("cleaned_data.csv", index=False)
    print("Cleaned data saved to 'cleaned_data.csv'")
