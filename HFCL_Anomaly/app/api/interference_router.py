# app/api/interference_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from app.utils.file_resolver import resolve_input_file_path

router = APIRouter()

INTERFERENCE_METRIC_FILE_MAP = {
    "interference report": {"type": "dynamic", "prefix": "ap_clients_5_min_daily"},
}

class InterferenceDetectionRequest(BaseModel):
    date: str = Field(..., description="Date for the dataset (e.g., '2025-05-15').")
    metric: str = Field("interference report", description="Metric for dataset resolution.")
    output_directory: str = Field("./reports/interference_detection/", description="Where to save CSV report.")
    contamination: float = Field(0.1, ge=0.0, le=0.5, description="Expected anomaly proportion for Isolation Forest.")
    n_estimators: int = Field(200, ge=50, description="Number of trees in Isolation Forest.")
    min_snr_threshold: float = Field(10.0, le=30.0, description="SNR threshold (dB) for boosting anomaly scores.")
    max_rssi_threshold: float = Field(-80.0, le=-30.0, description="RSSI threshold (dBm) for boosting anomaly scores.")
    boost_factor: float = Field(3.0, ge=1.0, description="Factor to amplify anomaly scores for critical cases.")

class InterferenceDetectionResponse(BaseModel):
    status: str
    message: str
    total_anomalies_found: int
    affected_aps: List[str]
    affected_channels: List[int]
    interference_report_csv: str

@router.post("/optimization/detect_interference", response_model=InterferenceDetectionResponse)
async def detect_interference(request: InterferenceDetectionRequest):
    try:
        # Load data
        input_file_path = resolve_input_file_path(request.date, request.metric, INTERFERENCE_METRIC_FILE_MAP)
        df = pd.read_csv(input_file_path)

        # Combine all Tx/Rx MCS fields into one total metric
        mcs_columns = [col for col in df.columns if 'Txmcs' in col or 'Rxmcs' in col]
        df['Total_MCS_Activity'] = df[mcs_columns].fillna(0).sum(axis=1)

        # Exclude rows with no traffic (MCS == 0)
        df_active = df[df['Total_MCS_Activity'] > 0].copy()
        if df_active.empty:
            raise HTTPException(status_code=404, detail="No rows with active MCS traffic found for analysis.")

        # Prepare features for Isolation Forest
        features = ['SNR', 'RSSI Strength', 'Total_MCS_Activity']
        df_features = df_active[features].fillna(0).copy()

        # Normalize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_features)

        # Fit Isolation Forest
        model = IsolationForest(n_estimators=request.n_estimators, contamination=request.contamination, random_state=42)
        model.fit(scaled_data)

        # Compute anomaly scores
        df_active['Anomaly_Score'] = -model.decision_function(scaled_data)

        # Boost anomaly scores for severe cases
        boost_mask = (
            (df_active['SNR'] < request.min_snr_threshold) &
            (df_active['RSSI Strength'] < request.max_rssi_threshold)
        )
        df_active.loc[boost_mask, 'Anomaly_Score'] *= request.boost_factor

        # Determine threshold
        threshold = df_active['Anomaly_Score'].mean() + 2 * df_active['Anomaly_Score'].std()
        df_active['Is_Anomaly'] = df_active['Anomaly_Score'] > threshold

        # Collect anomalies
        anomalies_df = df_active[df_active['Is_Anomaly']].copy()
        os.makedirs(request.output_directory, exist_ok=True)
        report_path = os.path.join(request.output_directory, "interference_anomalies_report.csv")
        anomalies_df.to_csv(report_path, index=False)

        # Prepare output
        affected_aps = sorted(anomalies_df['AP Name'].dropna().unique().tolist()) if 'AP Name' in anomalies_df.columns else []
        affected_channels = sorted(anomalies_df['Channel'].dropna().unique().tolist()) if 'Channel' in anomalies_df.columns else []

        return InterferenceDetectionResponse(
            status="success",
            message=f"Interference detection completed on {len(df_active)} active rows.",
            total_anomalies_found=len(anomalies_df),
            affected_aps=affected_aps,
            affected_channels=affected_channels,
            interference_report_csv=report_path
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during interference detection: {str(e)}")

