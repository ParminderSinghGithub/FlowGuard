# /ml_pipeline/feature_engineering.py
import pandas as pd

def generate_time_series_features(df):
    # Lag features: previous 1, 2, and 3 hours' traffic
    df['Lag_1'] = df['Vehicles_Normalized'].shift(1)
    df['Lag_2'] = df['Vehicles_Normalized'].shift(2)
    df['Lag_3'] = df['Vehicles_Normalized'].shift(3)
    
    # Rolling average feature
    df['Rolling_Mean_3'] = df['Vehicles_Normalized'].rolling(window=3).mean()
    
    # Drop rows with NaN values introduced by lag/rolling
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    # Load the cleaned data from the previous step
    cleaned_data_path = "cleaned_data.csv"  # Ensure this path matches your environment
    cleaned_data = pd.read_csv(cleaned_data_path)

    # Generate features
    feature_engineered_data = generate_time_series_features(cleaned_data)

    # Save the feature-engineered data
    feature_engineered_data.to_csv("feature_engineered_data.csv", index=False)
    print("Feature engineering completed and saved to 'feature_engineered_data.csv'")
