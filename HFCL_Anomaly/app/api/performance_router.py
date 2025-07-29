# app/api/performance_router.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os
import joblib # For saving/loading models
import pandas as pd
import numpy as np # Added for numerical operations
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # Added for PCA visualization
import matplotlib.pyplot as plt # Added for plotting
import time
import warnings # Added to suppress warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.ensemble._iforest")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=FutureWarning)


# Import the centralized file resolver
from app.utils.file_resolver import resolve_input_file_path

router = APIRouter()

# --- Model Persistence Configuration ---
MODEL_DIR = "./models"
PERFORMANCE_ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest_model_performance.joblib")

# Ensure model directory exists (can also be done in main.py)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Metric to File Mapping for Performance Router ---
PERFORMANCE_METRIC_FILE_MAP = {
    "performance anomalies": {"type": "dynamic", "prefix": "ap_clients_5_min_daily"},
}

# --- Request Models ---
class PerformanceAnomalyRequest(BaseModel):
    date: str = Field(
        ...,
        description="Date related to the dataset (e.g., 'May 15', '2025-05-15'). "
                    "For 'performance anomalies' metric, this date is for context only, "
                    "as the associated file is static in name. For dynamic metrics, it is used for file resolution."
    )
    metric: str = Field(
        "performance anomalies",
        description="Metric name to identify the dataset (e.g., 'performance anomalies', 'daily performance')."
    )
    output_directory: str = Field(
        "./reports/performance_anomalies/",
        description="Directory to save anomaly reports (CSV) and plots (PNG)."
    )
    generate_plots: bool = Field(
        True,
        description="If true, generates PCA anomaly plot and anomaly score timeline plot."
    )
    anomaly_score_threshold_multiplier: float = Field(
        2.0, ge=0.0,
        description="Multiplier for standard deviation to set the anomaly threshold."
    )
    
    # Training parameters (will be used only if model needs to be trained)
    if_contamination: Optional[float] = Field(
        0.1, ge=0.0, le=0.5,
        description="Contamination parameter for Isolation Forest (used if model needs training)."
    )
    if_n_estimators: Optional[int] = Field(
        100, ge=1,
        description="Number of estimators (trees) for Isolation Forest (used if model needs training)."
    )


# --- Response Models ---
class PerformanceAnomalyResponse(BaseModel):
    status: str = "success"
    message: str = "Anomaly detection completed."
    total_anomalies_found: int
    anomaly_report_csv: str = Field(
        ...,
        description="Path to the CSV report containing all original fields for anomalous rows, plus anomaly scores."
    )
    pca_plot_path: Optional[str] = None
    timeline_plot_path: Optional[str] = None
    dataset_shape: List[int]
    features_analyzed: List[str]
    model_trained_now: bool = Field(
        False,
        description="True if the model was trained during this API call, False if a pre-existing model was loaded."
    )
    training_duration_seconds: Optional[float] = Field(
        None,
        description="Time taken to train the model in seconds, if trained during this call."
    )

# --- Helper Function for Data Preprocessing (Used in both training and prediction) ---
def _preprocess_data(df: pd.DataFrame, features: List[str], is_training: bool = False) -> Dict[str, Any]:
    """
    Internal helper function to preprocess the data for anomaly detection.
    Handles Uptime conversion, fills missing values, and checks for constant columns.
    Returns the processed DataFrame, the list of actual numerical features used, and identifier columns.
    """
    df_processed = df.copy()

    # Robust Uptime conversion to seconds
    if 'Uptime' in df_processed.columns:
        def uptime_to_seconds(uptime_str):
            if pd.isna(uptime_str) or not isinstance(uptime_str, str):
                return np.nan
            try:
                parts = list(map(int, uptime_str.strip().split(':')))
                if len(parts) == 3: # HH:MM:SS
                    return parts[0] * 3600 + parts[1] * 60 + parts[2]
                elif len(parts) == 2: # MM:SS
                    return parts[0] * 60 + parts[1]
                else:
                    return np.nan
            except ValueError:
                return np.nan

        df_processed['Uptime (seconds)'] = df_processed['Uptime'].apply(uptime_to_seconds)
        
        # Impute missing Uptime (seconds) with median if in training, or if needed for prediction
        if 'Uptime (seconds)' in features:
            valid_uptimes = df_processed['Uptime (seconds)'].dropna()
            if not valid_uptimes.empty:
                median_uptime = valid_uptimes.median()
                df_processed['Uptime (seconds)'].fillna(median_uptime, inplace=True)
            elif is_training: # Only raise error during training if no valid uptime is found
                raise ValueError("No valid Uptime values found for median imputation during training.")


    # Select numerical features for processing based on 'features' parameter
    actual_numerical_features = [f for f in features if f in df_processed.columns]
    
    if not actual_numerical_features:
        raise ValueError("No valid numerical features found in dataset for anomaly detection.")

    # Keep identifier columns, including AP Mac and Timestamp if available
    identifier_columns = ['Client Mac', 'AP Mac', 'AP Name', 'Interface', 'Timestamp']
    identifier_columns = [col for col in identifier_columns if col in df_processed.columns]

    # Convert to numeric and handle missing values for numerical features
    for col in actual_numerical_features:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        if df_processed[col].isna().any():
            # Use mean imputation for missing numerical values
            mean_val = df_processed[col].mean()
            if pd.isna(mean_val) and is_training: # If mean itself is NaN due to all NaNs in training
                raise ValueError(f"Column '{col}' is entirely NaN. Cannot impute with mean during training.")
            df_processed[col].fillna(mean_val, inplace=True)
            
    # Check for zero-variance columns (only relevant for training where features are selected)
    if is_training:
        initial_features = list(actual_numerical_features) # Copy for iteration
        for col in initial_features:
            if df_processed[col].nunique() <= 1 or df_processed[col].var() < 1e-4:
                print(f"⚠️ Dropping constant or near-constant column during training: {col}")
                actual_numerical_features.remove(col)
        
        if not actual_numerical_features:
            raise ValueError("No variable numerical features remain after filtering during training.")

    # Check skewness for Uptime (seconds) and apply log transformation
    if 'Uptime (seconds)' in actual_numerical_features:
        skewness = df_processed['Uptime (seconds)'].skew()
        # A common threshold for skewness to consider log transformation is > 1 or > 2
        if skewness > 1:
            df_processed['Uptime (seconds)'] = np.log1p(df_processed['Uptime (seconds)'].clip(lower=0))
            print(f"Applied log transformation to Uptime (seconds) (skewness: {skewness:.2f})")
    
    # Ensure processed df only contains the features and identifiers that will be used
    df_final = df_processed[actual_numerical_features + identifier_columns].copy()

    return {
        "df_processed": df_final,
        "numerical_features_used": actual_numerical_features,
        "identifier_columns_used": identifier_columns
    }


# --- Helper Function for Training ---
async def _train_and_save_model(
    data_file_path: str,
    contamination: float,
    n_estimators: int,
    save_path: str
) -> Dict[str, Any]:
    """
    Internal helper function to train and save the Isolation Forest model.
    """
    start_time = time.time()
    try:
        df_train_raw = pd.read_csv(data_file_path)

        # Define initial features to be used for training.
        # This list defines what we *attempt* to use.
        initial_features_for_model = ['SNR', 'RSSI Strength', 'Uptime (seconds)']

        # Preprocess the training data
        processed_data_result = _preprocess_data(df_train_raw, initial_features_for_model, is_training=True)
        df_train_cleaned = processed_data_result['df_processed']
        features_for_training = processed_data_result['numerical_features_used']

        if df_train_cleaned[features_for_training].empty:
            raise ValueError("Cleaned training data is empty after dropping NaNs/filtering. Cannot train model.")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train_cleaned[features_for_training])

        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        model.fit(scaled_data)

        # Save model, scaler, and the *exact* list of features used for training
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'features': features_for_training
        }, save_path)
        end_time = time.time()
        
        return {
            "status": "success",
            "message": "Model trained and saved successfully.",
            "duration": (end_time - start_time),
            "features_used": features_for_training
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during model training: {str(e)}")


# --- API Endpoint ---

@router.post("/performance/detect_anomalies", response_model=PerformanceAnomalyResponse)
async def detect_performance_anomalies(request: PerformanceAnomalyRequest):
    """
    Detects network performance anomalies using dynamically selected input files.
    If a model is not trained, it will first train the Isolation Forest model using the resolved input data,
    then proceed with detection. Subsequent calls will use the saved model.
    """
    model_trained_now = False
    training_duration = None
    model = None
    scaler = None
    features_used_for_model = [] # Features the model was trained/will be trained on

    # Resolve the input file path using the centralized helper
    try:
        input_file_path = resolve_input_file_path(request.date, request.metric, PERFORMANCE_METRIC_FILE_MAP)
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions from the resolver

    # Check if a trained model exists
    if not os.path.exists(PERFORMANCE_ANOMALY_MODEL_PATH):
        print(f"No trained model found at {PERFORMANCE_ANOMALY_MODEL_PATH}. Initiating training...")
        model_trained_now = True
        
        # Train the model using the resolved input_file_path as training data
        train_result = await _train_and_save_model(
            data_file_path=input_file_path,
            contamination=request.if_contamination,
            n_estimators=request.if_n_estimators,
            save_path=PERFORMANCE_ANOMALY_MODEL_PATH
        )
        training_duration = train_result.get("duration")
        features_used_for_model = train_result.get("features_used", [])

        # Load the newly trained model to ensure consistent objects
        model_data = joblib.load(PERFORMANCE_ANOMALY_MODEL_PATH)
        model = model_data['model']
        scaler = model_data['scaler']
        features_used_for_model = model_data['features'] # Use the features confirmed by the training process
        
    else:
        print(f"Trained model found at {PERFORMANCE_ANOMALY_MODEL_PATH}. Loading model...")
        try:
            model_data = joblib.load(PERFORMANCE_ANOMALY_MODEL_PATH)
            model = model_data['model']
            scaler = model_data['scaler']
            features_used_for_model = model_data['features']
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error loading pre-trained model: {str(e)}. Consider deleting '{PERFORMANCE_ANOMALY_MODEL_PATH}' and retraining.")

    # Proceed with anomaly detection using the loaded/trained model
    try:
        df_original = pd.read_csv(input_file_path)
        
        # Preprocess input data for prediction using the same logic as training
        processed_input_result = _preprocess_data(df_original, features_used_for_model, is_training=False)
        df_input_for_prediction = processed_input_result['df_processed']
        numerical_features_actual = processed_input_result['numerical_features_used']
        identifier_columns_actual = processed_input_result['identifier_columns_used']

        # Ensure input data has the features the model was trained on
        missing_features = [f for f in features_used_for_model if f not in numerical_features_actual]
        if missing_features:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Input data is missing features required by the trained model: {', '.join(missing_features)}. Please ensure your input data matches the data used for training.")

        # Filter the DataFrame to only the numerical features required by the model and drop NaNs again
        # (Preprocessing already handled NaNs, this is a safety net if something slipped or if `features_used_for_model`
        # contains columns not in the current `df_input_for_prediction` after initial preprocessing)
        df_features_for_prediction_subset = df_input_for_prediction[numerical_features_actual].dropna()

        if df_features_for_prediction_subset.empty:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input data is empty or contains only NaNs after cleaning for prediction. Cannot perform anomaly detection.")

        # Get the original indices of the rows used for prediction, ensuring alignment
        original_indices_for_prediction = df_features_for_prediction_subset.index
        
        # Scale the data using the *trained* scaler
        scaled_data_detect = scaler.transform(df_features_for_prediction_subset)
        anomaly_scores_raw = model.decision_function(scaled_data_detect)
        
        # Invert scores for consistency (higher score = more anomalous)
        anomaly_scores = -anomaly_scores_raw

        # Map anomaly scores back to the original DataFrame
        df_original['Anomaly_Score'] = pd.Series(anomaly_scores, index=original_indices_for_prediction).reindex(df_original.index)
        df_original['Anomaly_Score'] = df_original['Anomaly_Score'].fillna(np.nan) # Fill NaN for rows not used due to missing features


        # Define strict thresholds for "very low" and "spikes" based on the original data statistics
        # (These are *not* from the scaled data or model features, but from the raw data for interpretation)
        thresholds = {}
        # Ensure that these columns actually exist in the original_df and are numeric
        for col in ['SNR', 'RSSI Strength', 'Uptime (seconds)', 'Data Uploaded (MB)', 'Data Downloaded (MB)']:
            if col in df_original.columns and pd.api.types.is_numeric_dtype(df_original[col]):
                if col in ['SNR', 'RSSI Strength', 'Uptime (seconds)']:
                    # Use a combination of statistical and domain-specific thresholds
                    thresholds[col] = df_original[col].quantile(0.10)  # Bottom 10% for very low
                    if col == 'SNR': thresholds[col] = min(thresholds[col], 10)  # SNR < 10 dB
                    if col == 'RSSI Strength': thresholds[col] = min(thresholds[col], -80)  # RSSI < -80 dBm
                    # For Uptime, use the *original* uptime seconds for thresholding, not the log-transformed
                    if col == 'Uptime (seconds)': thresholds[col] = min(thresholds[col], 600)  # Uptime < 10 min
                elif col in ['Data Uploaded (MB)', 'Data Downloaded (MB)']:
                    mean = df_original[col].mean()
                    std = df_original[col].std()
                    thresholds[col] = mean + 3 * std  # Extreme spikes: >3 std deviations
        
        print("Strict Thresholds for Anomaly Detection:")
        for col, thresh in thresholds.items():
            print(f"{col}: {thresh:.2f}")

        # Priority-based scoring adjustment
        # Apply this based on the original data, then update the anomaly score series
        spike_driven_flags = pd.Series(False, index=df_original.index)
        temp_anomaly_scores_adjusted = df_original['Anomaly_Score'].copy()

        for i, original_row in df_original.iterrows():
            if pd.isna(original_row['Anomaly_Score']): # Skip rows that didn't have valid numerical features for scoring
                continue

            is_very_low_snr = 'SNR' in thresholds and original_row.get('SNR', np.inf) <= thresholds['SNR']
            is_very_low_rssi = 'RSSI Strength' in thresholds and original_row.get('RSSI Strength', np.inf) <= thresholds['RSSI Strength']
            is_very_low_uptime = 'Uptime (seconds)' in thresholds and original_row.get('Uptime (seconds)', np.inf) <= thresholds['Uptime (seconds)']
            is_high_upload = 'Data Uploaded (MB)' in thresholds and original_row.get('Data Uploaded (MB)', -np.inf) >= thresholds['Data Uploaded (MB)']
            is_high_download = 'Data Downloaded (MB)' in thresholds and original_row.get('Data Downloaded (MB)', -np.inf) >= thresholds['Data Downloaded (MB)']

            # Check if all low conditions and at least one high condition are met
            if is_very_low_snr and is_very_low_rssi and is_very_low_uptime and (is_high_upload or is_high_download):
                temp_anomaly_scores_adjusted.loc[i] *= 3.0  # Amplify score for priority cases
                spike_driven_flags.loc[i] = True
            else:
                temp_anomaly_scores_adjusted.loc[i] *= 0.1 # Reduce score for non-priority cases

        df_original['Anomaly_Score_Adjusted'] = temp_anomaly_scores_adjusted
        df_original['Spike_Driven'] = spike_driven_flags

        # Apply anomaly detection logic based on threshold_multiplier on the ADJUSTED scores
        valid_adjusted_scores = df_original['Anomaly_Score_Adjusted'].dropna()
        if valid_adjusted_scores.empty:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No valid adjusted anomaly scores generated for thresholding after prediction.")

        mean_score_adj = valid_adjusted_scores.mean()
        std_dev_score_adj = valid_adjusted_scores.std()
        anomaly_threshold_final = mean_score_adj + request.anomaly_score_threshold_multiplier * std_dev_score_adj
        
        df_original['Is_Anomaly'] = (df_original['Anomaly_Score_Adjusted'] > anomaly_threshold_final).fillna(False)

        # --- Generate the anomaly report CSV with ALL original fields ---
        anomalies_df = df_original[df_original['Is_Anomaly']].sort_values(by='Anomaly_Score_Adjusted', ascending=False).copy()
        
        # Create output directory if it doesn't exist
        os.makedirs(request.output_directory, exist_ok=True)
        report_csv_path = os.path.join(request.output_directory, "performance_anomalies_report.csv")
        anomalies_df.to_csv(report_csv_path, index=False)


        # --- PCA for 2D visualization ---
        pca_plot_path = None
        if request.generate_plots and not df_features_for_prediction_subset.empty:
            try:
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(scaled_data_detect) # Use the scaled data that was used for prediction
                df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=original_indices_for_prediction)
                
                # Merge 'Is_Anomaly' and 'Spike_Driven' back to df_pca based on index
                df_pca = pd.merge(df_pca, df_original[['Is_Anomaly', 'Spike_Driven']], left_index=True, right_index=True, how='left')
                
                plt.figure(figsize=(12, 8))
                plt.scatter(df_pca.loc[~df_pca['Is_Anomaly'], 'PC1'], df_pca.loc[~df_pca['Is_Anomaly'], 'PC2'],
                            c='blue', s=50, alpha=0.6, label='Normal')
                plt.scatter(df_pca.loc[df_pca['Is_Anomaly'] & ~df_pca['Spike_Driven'], 'PC1'],
                            df_pca.loc[df_pca['Is_Anomaly'] & ~df_pca['Spike_Driven'], 'PC2'],
                            c='red', s=100, marker='x', label='Anomaly (Non-Spike)')
                plt.scatter(df_pca.loc[df_pca['Spike_Driven'], 'PC1'], df_pca.loc[df_pca['Spike_Driven'], 'PC2'],
                            c='orange', s=150, marker='^', label='Spike-Driven Anomaly')
                plt.title('Isolation Forest Anomalies (PCA Projection)')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.legend()
                pca_plot_path = os.path.join(request.output_directory, 'isolation_forest_anomalies_pca.png')
                plt.savefig(pca_plot_path)
                plt.close()
                print(f"PCA Anomaly plot saved to: {pca_plot_path}")
            except Exception as e:
                print(f"Warning: Could not generate PCA plot due to error: {e}")
                pca_plot_path = None


        # --- Anomaly timeline plot ---
        timeline_plot_path = None
        if request.generate_plots and 'Timestamp' in df_original.columns:
            try:
                # Ensure Timestamp is datetime for proper plotting
                df_original['Timestamp_dt'] = pd.to_datetime(df_original['Timestamp'], errors='coerce')
                # Filter out rows where Timestamp could not be parsed
                df_plot = df_original.dropna(subset=['Timestamp_dt', 'Anomaly_Score_Adjusted']).sort_values('Timestamp_dt')

                plt.figure(figsize=(15, 7))
                plt.plot(df_plot['Timestamp_dt'], df_plot['Anomaly_Score_Adjusted'], label='Anomaly Score', color='blue', alpha=0.7)
                plt.scatter(df_plot[df_plot['Is_Anomaly'] & ~df_plot['Spike_Driven']]['Timestamp_dt'],
                            df_plot[df_plot['Is_Anomaly'] & ~df_plot['Spike_Driven']]['Anomaly_Score_Adjusted'],
                            color='red', s=50, label='Anomaly (Non-Spike)')
                plt.scatter(df_plot[df_plot['Spike_Driven']]['Timestamp_dt'],
                            df_plot[df_plot['Spike_Driven']]['Anomaly_Score_Adjusted'],
                            color='orange', s=100, marker='^', label='Spike-Driven Anomaly')
                plt.axhline(y=anomaly_threshold_final, color='green', linestyle='--', label=f'Threshold ({anomaly_threshold_final:.2f})')
                plt.title('Anomaly Scores Over Time (Isolation Forest)')
                plt.xlabel('Timestamp')
                plt.ylabel('Adjusted Anomaly Score')
                plt.legend()
                plt.tight_layout() # Adjust layout to prevent labels overlapping
                timeline_plot_path = os.path.join(request.output_directory, 'anomaly_timeline_isolation_forest.png')
                plt.savefig(timeline_plot_path)
                plt.close()
                print(f"Anomaly timeline plot saved to: {timeline_plot_path}")
            except Exception as e:
                print(f"Warning: Could not generate timeline plot due to error: {e}")
                timeline_plot_path = None

        
        return PerformanceAnomalyResponse(
            status="success",
            message=f"Anomaly detection completed. Model {'trained' if model_trained_now else 'loaded'} and used.",
            total_anomalies_found=anomalies_df.shape[0],
            anomaly_report_csv=report_csv_path,
            pca_plot_path=pca_plot_path,
            timeline_plot_path=timeline_plot_path,
            dataset_shape=list(df_original.shape), # Return shape of original df
            features_analyzed=features_used_for_model,
            model_trained_now=model_trained_now,
            training_duration_seconds=training_duration
        )
    except FileNotFoundError: # This will be handled by resolve_input_file_path now
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Input data file not found after resolution.")
    except HTTPException as http_exc: # Re-raise HTTPExceptions from inner logic
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during anomaly detection: {str(e)}")
