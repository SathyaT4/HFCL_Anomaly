import pandas as pd
import numpy as np
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from datetime import timedelta

try:
    from app.utils.file_resolver import resolve_input_file_path
except ImportError:
    def resolve_input_file_path(date: str, metric_name: str, file_map: Dict):
        return f"ap_clients_5_min_daily_{date.replace('-', '_')}.csv"

router = APIRouter()

CONNECTION_DROP_METRIC_FILE_MAP = {
    "connection drops": {"type": "dynamic", "prefix": "ap_clients_5_min_daily"},
}

class ConnectionDropRequest(BaseModel):
    date: str
    output_directory: str = "./reports/connection_drops/"
    min_connection_duration_threshold_seconds: Optional[int] = 30
    assoc_threshold: int = 3
    assoc_window_seconds: int = 60

class ConnectionDropEventSummary(BaseModel):
    client_mac: str
    ap_name: str
    ssid: str
    interface: str
    connection_duration_seconds: Optional[int] = None
    association_time: Optional[str] = None
    disassociation_time: Optional[str] = None
    anomaly_score: Optional[float] = None
    ml_anomaly_detected: bool = False
    ml_recommendation: str = ""
    rapid_assoc_anomaly: bool = False

class ConnectionDropResponse(BaseModel):
    status: str
    message: str
    total_disconnection_events: int
    unique_clients_affected: int
    total_ml_anomalies: int
    client_disconnection_counts: Dict[str, int]
    disconnection_events: List[ConnectionDropEventSummary]
    # connection_drops_report_csv: str
    # resolved_input_file: str

@router.post("/connection_drops/analyze_drops", response_model=ConnectionDropResponse)
async def analyze_drops(request: ConnectionDropRequest):
    try:
        input_file_path = resolve_input_file_path(request.date, "connection drops", CONNECTION_DROP_METRIC_FILE_MAP)
        print(f"[INFO] Loading CSV from {input_file_path}")
        df = pd.read_csv(input_file_path)

        # Parse timestamps
        df['Association Time_dt'] = pd.to_datetime(df['Association Time'], errors='coerce', dayfirst=True)
        df['Disassociation Time_dt'] = pd.to_datetime(df['Disassociation Time'], errors='coerce', dayfirst=True)
        df = df[df['Association Time_dt'].notna()].copy()
        print(f"[INFO] Valid association time rows: {len(df)}")

        # Handle missing disassociation time (set as string "Ongoing Session")
        df['Disassociation Time'] = df['Disassociation Time'].fillna("Ongoing Session").astype(str)

        # Calculate connection durations (0 if ongoing)
        df['connection_duration_seconds'] = (df['Disassociation Time_dt'] - df['Association Time_dt']).dt.total_seconds()
        df.loc[df['Disassociation Time_dt'].isna(), 'connection_duration_seconds'] = 0

        # Sort by client, AP, time
        df = df.sort_values(['Client Mac', 'AP Mac', 'Association Time_dt']).reset_index(drop=True)

        # Detect anomalies (3+ reconnects within 60 seconds)
        rapid_flags = [False] * len(df)
        anomaly_logs = []

        assoc_threshold = request.assoc_threshold or 3
        assoc_window_seconds = request.assoc_window_seconds or 60

        for (client, ap), group in df.groupby(['Client Mac', 'AP Mac']):
            times = group['Association Time_dt'].tolist()
            idxs = group.index.tolist()

            # Filter duplicate association times (polling)
            filtered_times, filtered_idxs = [], []
            for i, t in enumerate(times):
                if not filtered_times or t != filtered_times[-1]:
                    filtered_times.append(t)
                    filtered_idxs.append(idxs[i])

            # Sliding window detection
            for i in range(len(filtered_times)):
                window_start = filtered_times[i]
                j = i
                while j < len(filtered_times) and (filtered_times[j] - window_start).total_seconds() <= assoc_window_seconds:
                    j += 1
                if (j - i) >= assoc_threshold:
                    for k in range(i, j):
                        flag_idx = filtered_idxs[k]
                        rapid_flags[df.index.get_loc(flag_idx)] = True
                    anomaly_logs.append(
                        f"[ANOMALY] Client {client} @ AP {ap}: {j - i} reconnects "
                        f"from {filtered_times[i]} to {filtered_times[j-1]}"
                    )

        # Keep only anomalies
        df['rapid_assoc_anomaly'] = rapid_flags
        anomalies_df = df[df['rapid_assoc_anomaly']].copy()

        print(f"[INFO] Total anomalies found: {len(anomalies_df)}")
        for log in anomaly_logs:
            print(log)

        # Save anomalies-only CSV
        os.makedirs(request.output_directory, exist_ok=True)
        report_path = os.path.join(request.output_directory, "connection_drops_report.csv")
        anomalies_df.to_csv(report_path, index=False)
        print(f"[INFO] Anomalies-only report saved: {report_path}")

        # Build JSON response (ensure all strings)
        events = [
            ConnectionDropEventSummary(
                client_mac=row['Client Mac'],
                ap_name=row['AP Name'],
                ssid=row['SSID'],
                interface=row['Interface'],
                connection_duration_seconds=row.get('connection_duration_seconds', 0),
                association_time=str(row['Association Time']) if not pd.isna(row['Association Time']) else "",
                disassociation_time=str(row['Disassociation Time']) if not pd.isna(row['Disassociation Time']) else "",
                anomaly_score=None,
                ml_anomaly_detected=False,
                ml_recommendation="Frequent reconnect anomaly",
                rapid_assoc_anomaly=True
            )
            for _, row in anomalies_df.iterrows()
        ]

        return ConnectionDropResponse(
            status="success",
            message=f"Rule-based detection complete. {len(anomalies_df)} anomalies found.",
            total_disconnection_events=len(anomalies_df),
            unique_clients_affected=anomalies_df['Client Mac'].nunique(),
            total_ml_anomalies=0,
            client_disconnection_counts=anomalies_df['Client Mac'].value_counts().to_dict(),
            disconnection_events=events,
           # connection_drops_report_csv=report_path,
           # resolved_input_file=input_file_path
        )

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")