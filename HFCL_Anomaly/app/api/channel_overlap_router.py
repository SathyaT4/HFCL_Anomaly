import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os

# Import the centralized file resolver
from app.utils.file_resolver import resolve_input_file_path

router = APIRouter()

# --- Metric to File Mapping for Channel Optimization Router ---
CONNECTION_DROP_METRIC_FILE_MAP = { # Renamed to CONNECTION_DROP_METRIC_FILE_MAP as it maps to channel overlap report
    "channel overlap report": {"type": "dynamic", "prefix": "ap_clients_5_min_daily"},
}

# ---------- Helper Functions ----------
def channels_overlap(ch1, ch2, band='2.4GHz'):
    """Return True if two channels overlap given their band rules."""
    if band == '2.4GHz':
        # Channels spaced 5 MHz apart; 20 MHz wide (1,6,11 are considered safe non-overlapping)
        # Based on the original user's logic, channels are considered overlapping
        # unless they are distinct members of the commonly non-overlapping set {1, 6, 11}.
        non_overlapping_20MHz_channels = {1, 6, 11}
        
        if (ch1 in non_overlapping_20MHz_channels and 
            ch2 in non_overlapping_20MHz_channels and 
            ch1 != ch2):
            return False # These specific pairs (e.g., 1 and 6) do not overlap
        
        return True # All other 2.4GHz channel pairs are considered overlapping for this definition

    elif band == '5GHz':
        # Treat 5GHz channels typically with 20 MHz spacing.
        # Overlap if center frequencies are less than 20 MHz apart (or closer than one channel width).
        return abs(ch1 - ch2) < 20
    return False

def detect_channel_band(channel):
    """Determine if a channel belongs to 2.4GHz or 5GHz band."""
    if 1 <= channel <= 14:
        return '2.4GHz'
    elif 36 <= channel <= 196: # Extended range for 5GHz channels
        return '5GHz'
    return 'Unknown'


# --- Request Models ---
class ChannelOptimizationRequest(BaseModel):
    date: str = Field(
        ...,
        description="Date for the dataset (e.g., 'May 15', '2025-05-15'). Used for dynamic file resolution."
    )
    metric: str = Field(
        "channel overlap report",
        description="Metric name to identify the dataset (e.g., 'channel overlap report')."
    )
    output_directory: str = Field(
        "./reports/channel_optimization/",
        description="Directory to save channel optimization reports."
    )
    # ML model parameters
    contamination_level: float = Field(
        0.05, ge=0.0, le=0.5,
        description="Expected proportion of outliers in the data for Isolation Forest anomaly detection."
    )

# --- Response Models ---
class ChannelIssueSummary(BaseModel):
    ap_name: str
    ssid: str
    channel: Optional[int]
    interface: str
    rule_based_overlap_detected: bool = False
    overlapping_channels: List[str] = [] # e.g., ["36-40 (5GHz)", "1-6 (2.4GHz)"]
    ml_anomaly_detected: bool = False
    anomaly_score: Optional[float] = None # Lower score indicates more anomalous
    avg_snr: Optional[float] = None
    avg_rssi: Optional[float] = None
    client_count: Optional[int] = None
    recommendation: str = ""

class ChannelOptimizationResponse(BaseModel):
    status: str = "success"
    message: str = "Channel optimization analysis completed."
    total_rule_based_overlaps: int
    total_ml_anomalies: int
    channel_issues: List[ChannelIssueSummary]
    # channel_optimization_report_csv: str = Field(
    #     ...,
    #     description="Path to the CSV report containing detailed channel issue findings."
    # )
    # resolved_input_file: str = Field(
    #     ...,
    #     description="The actual file path used for analysis."
    # )

@router.post("/channel_optimization/analyze_channels", response_model=ChannelOptimizationResponse)
async def analyze_channels(request: ChannelOptimizationRequest):
    """
    Analyzes channel usage within Access Points to detect rule-based overlaps
    and identifies anomalous channel performance using an ML model (Isolation Forest).
    """
    try:
        metric_const = 'channel overlap report'
        # Resolve the input file path using the centralized helper
        input_file_path = resolve_input_file_path(request.date, metric_const, CONNECTION_DROP_METRIC_FILE_MAP)
        
        df = pd.read_csv(input_file_path)

        # Ensure Channel, RSSI Strength, SNR are numeric
        df['Channel'] = pd.to_numeric(df['Channel'], errors='coerce')
        df['RSSI Strength'] = pd.to_numeric(df['RSSI Strength'], errors='coerce')
        df['SNR'] = pd.to_numeric(df['SNR'], errors='coerce')

        # Drop rows with NaN in critical columns for analysis
        df_cleaned = df.dropna(subset=['AP Name', 'SSID', 'Channel', 'Interface', 'RSSI Strength', 'SNR']).copy()

        if df_cleaned.empty:
            os.makedirs(request.output_directory, exist_ok=True)
            report_csv_path = os.path.join(request.output_directory, "channel_optimization_report.csv")
            pd.DataFrame().to_csv(report_csv_path, index=False) # Create empty CSV

            return ChannelOptimizationResponse(
                status="success",
                message="No sufficient data for channel analysis after cleaning.",
                total_rule_based_overlaps=0,
                total_ml_anomalies=0,
                channel_issues=[],
                channel_optimization_report_csv=report_csv_path,
                resolved_input_file=input_file_path
            )

        # --- 1. Rule-Based Channel Overlap Detection ---
        rule_based_results = []
        # Group by AP, SSID, and Interface to find channels used within that specific broadcast
        for (ap_name, ssid, interface), group in df_cleaned.groupby(['AP Name', 'SSID', 'Interface']):
            unique_channels = group['Channel'].unique().tolist()
            unique_channels.sort() # Sort for consistent overlap reporting

            overlaps = []
            for i in range(len(unique_channels)):
                for j in range(i + 1, len(unique_channels)):
                    ch1 = unique_channels[i]
                    ch2 = unique_channels[j]
                    band1 = detect_channel_band(ch1)
                    band2 = detect_channel_band(ch2)

                    if band1 != 'Unknown' and band1 == band2 and channels_overlap(ch1, ch2, band1):
                        overlaps.append(f"{int(ch1)}-{int(ch2)} ({band1})")
            
            # If any overlaps are detected for this AP/SSID/Interface combination
            if overlaps:
                # Get average SNR/RSSI for clients associated with this specific AP/SSID/Interface
                avg_snr_group = group['SNR'].mean()
                avg_rssi_group = group['RSSI Strength'].mean()
                client_count_group = group['Client Mac'].nunique()

                # For rule-based overlaps, the issue is with the *combination* of channels on this specific AP/SSID/Interface.
                # We will create an entry for each unique channel involved in an overlap for reporting.
                for channel_in_group in unique_channels:
                    rule_based_results.append({
                        'ap_name': ap_name,
                        'ssid': ssid,
                        'channel': int(channel_in_group),
                        'interface': interface,
                        'rule_based_overlap_detected': True,
                        'overlapping_channels': overlaps,
                        'avg_snr': round(avg_snr_group, 2) if pd.notna(avg_snr_group) else None,
                        'avg_rssi': round(avg_rssi_group, 2) if pd.notna(avg_rssi_group) else None,
                        'client_count': client_count_group,
                        'recommendation': f"Potential rule-based channel overlap detected: {', '.join(overlaps)}. Consider reconfiguring channels on {ap_name} for {ssid} ({interface})."
                    })
            
        # Create a DataFrame for rule-based results
        if rule_based_results:
            rule_based_df = pd.DataFrame(rule_based_results)
            # Drop duplicates if multiple channels in the same group were flagged (e.g., if 1-6 and 1-11 overlap, channel 1 would appear twice)
            rule_based_df = rule_based_df.drop_duplicates(subset=['ap_name', 'ssid', 'channel', 'interface'])
        else:
            # If no rule-based overlaps, create an empty DataFrame with the expected columns
            rule_based_df = pd.DataFrame(columns=[
                'ap_name', 'ssid', 'channel', 'interface', 
                'rule_based_overlap_detected', 'overlapping_channels', 'recommendation'
            ])


        # --- 2. ML-Based Anomaly Detection for Channel Performance ---
        # Features for Isolation Forest: Average SNR, Average RSSI, Client Count per channel/AP/SSID/Interface
        ml_features_df = df_cleaned.groupby(['AP Name', 'SSID', 'Channel', 'Interface']).agg(
            avg_snr=('SNR', 'mean'),
            avg_rssi=('RSSI Strength', 'mean'),
            client_count=('Client Mac', 'nunique')
        ).reset_index()

        # Handle potential NaNs in features for Isolation Forest by filling with median
        features_for_model = ml_features_df[['avg_snr', 'avg_rssi', 'client_count']].copy()
        features_for_model = features_for_model.fillna(features_for_model.median())
        
        # Check if we have enough data to train Isolation Forest and if features are not all NaN
        if len(features_for_model) > 1 and not features_for_model.isnull().all().all():
            model = IsolationForest(contamination=request.contamination_level, random_state=42)
            model.fit(features_for_model)
            ml_features_df['anomaly_score'] = model.decision_function(features_for_model)
            # anomaly_score < 0 indicates an outlier
            ml_features_df['ml_anomaly_detected'] = (ml_features_df['anomaly_score'] < 0)
        else:
            ml_features_df['anomaly_score'] = np.nan
            ml_features_df['ml_anomaly_detected'] = False
        
        # --- 3. Combine Results ---
        # Start with all unique AP/SSID/Channel/Interface combinations observed in the cleaned data
        all_channel_combos = df_cleaned[['AP Name', 'SSID', 'Channel', 'Interface']].drop_duplicates().copy()
        
        # Merge with ML findings (left merge to keep all unique combos)
        final_df = pd.merge(all_channel_combos, ml_features_df, on=['AP Name', 'SSID', 'Channel', 'Interface'], how='left')
        
        # Merge with Rule-Based findings
        # Use the prepared rule_based_df directly as it's guaranteed to have the columns
        final_df = pd.merge(final_df, rule_based_df, 
                            left_on=['AP Name', 'SSID', 'Channel', 'Interface'], 
                            right_on=['ap_name', 'ssid', 'channel', 'interface'],
                            how='left', suffixes=('', '_rule'))
        
        # Clean up temporary merge columns
        final_df.drop(columns=['ap_name', 'ssid', 'channel', 'interface'], inplace=True, errors='ignore') # Removed _rule suffix from drop
        
        # Fill NaN values for flags and lists (for combinations not flagged by rule-based or ML)
        final_df['rule_based_overlap_detected'] = final_df['rule_based_overlap_detected'].fillna(False)
        final_df['overlapping_channels'] = final_df['overlapping_channels'].apply(lambda x: x if isinstance(x, list) else [])
        final_df['ml_anomaly_detected'] = final_df['ml_anomaly_detected'].fillna(False)
        final_df['anomaly_score'] = final_df['anomaly_score'].fillna(np.nan) # Keep NaN if no ML model was run or score wasn't applicable
        final_df['recommendation'] = final_df['recommendation'].fillna("") # Fill empty string for no recommendation yet
        
        # Add specific recommendations for ML anomalies if no rule-based recommendation exists
        final_df.loc[
            final_df['ml_anomaly_detected'] & (final_df['recommendation'] == ""),
            'recommendation'
        ] = "ML model detected anomalous channel performance for this configuration (AP, SSID, Channel, Interface). Investigate 'Avg_SNR' and 'Avg_RSSI' values which might indicate hidden interference or poor signal."
        
        # Prepare for JSON response and count totals
        channel_issues = []
        total_rule_based_overlaps = 0
        total_ml_anomalies = 0

        # Iterate over combined results to populate the Pydantic list
        # Only include rows where either rule-based or ML anomaly is detected
        problem_channels_df = final_df[
            (final_df['rule_based_overlap_detected'] == True) | 
            (final_df['ml_anomaly_detected'] == True)
        ].copy()


        for _, row in problem_channels_df.iterrows():
            if row['rule_based_overlap_detected']:
                total_rule_based_overlaps += 1
            if row['ml_anomaly_detected']:
                total_ml_anomalies += 1
                
            issue_summary = ChannelIssueSummary(
                ap_name=row['AP Name'],
                ssid=row['SSID'],
                channel=int(row['Channel']) if pd.notna(row['Channel']) else None,
                interface=row['Interface'],
                rule_based_overlap_detected=bool(row['rule_based_overlap_detected']),
                overlapping_channels=row['overlapping_channels'],
                ml_anomaly_detected=bool(row['ml_anomaly_detected']),
                anomaly_score=round(row['anomaly_score'], 4) if pd.notna(row['anomaly_score']) else None,
                avg_snr=round(row['avg_snr'], 2) if pd.notna(row['avg_snr']) else None,
                avg_rssi=round(row['avg_rssi'], 2) if pd.notna(row['avg_rssi']) else None,
                client_count=int(row['client_count']) if pd.notna(row['client_count']) else None,
                recommendation=row['recommendation']
            )
            channel_issues.append(issue_summary)

        # Create output directory
        os.makedirs(request.output_directory, exist_ok=True)
        report_csv_path = os.path.join(request.output_directory, "channel_optimization_report.csv")
        
        # Select relevant columns for CSV output
        csv_output_df = problem_channels_df[[
            'AP Name', 'SSID', 'Channel', 'Interface', 
            'rule_based_overlap_detected', 'overlapping_channels',
            'ml_anomaly_detected', 'anomaly_score',
            'avg_snr', 'avg_rssi', 'client_count', 'recommendation'
        ]].copy()
        
        # Ensure 'overlapping_channels' is stringified for CSV
        csv_output_df['overlapping_channels'] = csv_output_df['overlapping_channels'].apply(lambda x: "; ".join(x) if isinstance(x, list) else "")
        csv_output_df.to_csv(report_csv_path, index=False)

        return ChannelOptimizationResponse(
            status="success",
            message="Channel optimization analysis completed.",
            total_rule_based_overlaps=total_rule_based_overlaps,
            total_ml_anomalies=total_ml_anomalies,
            channel_issues=channel_issues,
            # channel_optimization_report_csv=report_csv_path,
            # resolved_input_file=input_file_path
        )
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Input data file not found after resolution.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during channel optimization: {str(e)}")