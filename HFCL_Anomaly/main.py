import datetime
from typing import Any, Dict
from fastapi import APIRouter, FastAPI, HTTPException, Query
import os
import pandas as pd
import uvicorn
import httpx
# Import your API routers
from app.api.performance_router import router as performance_router
from app.api.interference_router import router as interference_router
from app.api.connection_drop_router import router as connection_drop_router
from app.api.channel_overlap_router import router as channel_overlap_router
from app.api.get_info import router as get_info_router
from app.api.plot_chart import router as plot_chart_router
from app.utils.file_resolver import BASE_DATA_REPORTS_DIR
from app.api.get_all import router as unified_anomaly_router

router = APIRouter()

# Initialize FastAPI app
app = FastAPI(
    title="Network Monitoring & Optimization API",
    description="API for detecting anomalies, interference, connection issues, and channel overlaps in network data.",
    version="1.0.0"
)

# Include API routers
app.include_router(performance_router, prefix="/api")
app.include_router(interference_router, prefix="/api")
app.include_router(connection_drop_router, prefix="/api")
app.include_router(channel_overlap_router, prefix="/api")
app.include_router(unified_anomaly_router, prefix="/api")
app.include_router(get_info_router, prefix="/api")
app.include_router(plot_chart_router, prefix="/api")

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Network Monitoring & Optimization API!",
        "documentation": "Visit /docs for OpenAPI (Swagger UI) documentation.",
        "redoc_documentation": "Visit /redoc for ReDoc documentation."
    }

# ------------------ Unified Anomaly API ------------------

REPORT_CONFIGS = {
    "/performance/detect_anomalies": {
        "module_name": "Performance",
        "report_filename": "performance_anomalies_report.csv",
        "report_directory": "./reports/performance_anomalies/"
    },
    "/optimization/detect_channel_overlap": {
        "module_name": "ChannelOverlap",
        "report_filename": "channel_overlap_report.csv",
        "report_directory": "./reports/channel_overlap/"
    },
    "/connection_drops/analyze_drops": {
        "module_name": "ConnectionDrops",
        "report_filename": "connection_drops_report.csv",
        "report_directory": "./reports/connection_drops/"
    },
    "/optimization/detect_interference": {
        "module_name": "Interference",
        "report_filename": "interference_anomalies_report.csv",
        "report_directory": "./reports/interference_detection/"
    }
}

# Define the directory and filename for the unified anomaly report
UNIFIED_REPORT_DIR = "./reports/unified_anomalies/"
UNIFIED_REPORT_FILENAME = "unified_anomaly_report.csv"


def read_and_standardize_report(file_path: str, source_module: str) -> pd.DataFrame:
    """
    Reads a CSV report, adds a source module column, and standardizes common column names.
    Fills NaN values with None for proper JSON serialization.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Report file not found at {file_path}. Skipping.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        # Add a 'source_module' column to identify the origin of the anomaly
        df['source_module'] = source_module

        # Standardize common column names to ensure consistency across merged reports
        # You may need to adjust these based on the actual column names from your specific reports.
        df.rename(columns={
            'AP Name': 'ap_name',
            'Client Mac': 'client_mac',
            'Channel': 'channel',
            'Timestamp': 'timestamp', # Assuming a common timestamp column if it exists
            'Host Name': 'host_name',
            'Device Type': 'device_type',
            'Device OS': 'device_os',
            'Anomaly_Score': 'anomaly_score',
            'Is_Anomaly': 'is_anomaly',
            'ml_anomaly_detected': 'ml_anomaly_detected',
            'ml_recommendation': 'ml_recommendation',
            'Total_MCS_Activity': 'total_mcs_activity',
            'SNR': 'snr',
            'RSSI Strength': 'rssi_strength'
            # Add more renames as needed for other reports
        }, inplace=True)

        # Convert column names to snake_case for consistency in JSON output
        df.columns = [col.replace(' ', '_').lower() for col in df.columns]

        # Fill any remaining NaN values with None for proper JSON serialization
        df = df.where(pd.notnull(df), None)
        return df
    except pd.errors.EmptyDataError:
        print(f"Warning: {file_path} is empty. Skipping.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading or processing {file_path}: {e}")
        return pd.DataFrame()


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file if it exists, else return empty DataFrame."""
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# --- Unified GET endpoint ---
@router.get("/api/anomalies/summary", response_model=Dict[str, Any])
async def get_unified_anomalies(
    call_all: bool = Query(False, description="If true, triggers all anomaly APIs (POST) before merging.")
):
    """
    Unified GET endpoint that optionally triggers all anomaly detection submodules (via POST)
    and then reads, merges, and summarizes their generated reports.
    A unified CSV report of all anomalies is also created/rewritten.
    """
    base_url = "http://0.0.0.0:8989/api" # Base URL where your FastAPI app is running
    current_date_str = datetime.date.today().strftime('%Y-%m-%d')

    # 1. Trigger all anomaly endpoints if 'call_all' is True
    if call_all:
        print("Triggering all anomaly detection submodules...")
        async with httpx.AsyncClient() as client:
            for ep, config in REPORT_CONFIGS.items():
                try:
                    # Ensure the output directory for the specific report exists before the POST call
                    os.makedirs(config["report_directory"], exist_ok=True)

                    # Construct a basic payload for the POST request.
                    # This might need to be customized for each specific POST endpoint
                    # if they have different required fields beyond 'date'.
                    post_data = {"date": current_date_str}

                    # Special handling for detect_interference which needs 'output_directory'
                    if ep == "/optimization/detect_interference":
                        post_data.update({
                            "output_directory": config["report_directory"],
                            "metric": "interference report", # Default value from InterferenceDetectionRequest
                            "contamination": 0.1,
                            "n_estimators": 200,
                            "min_snr_threshold": 10.0,
                            "max_rssi_threshold": -80.0,
                            "boost_factor": 3.0
                        })
                    # Add similar 'elif' blocks for other specific POST endpoint payloads if needed

                    resp = await client.post(f"{base_url}{ep}", json=post_data, timeout=120.0) # Increased timeout for long-running tasks
                    if resp.status_code == 200:
                        print(f"Successfully triggered {ep}. Response: {resp.json().get('message', 'No message')}")
                    else:
                        print(f"Warning: {ep} returned HTTP {resp.status_code}. Response: {resp.text}")
                except httpx.RequestError as e:
                    print(f"Failed to connect to {ep} (Is the service running?): {e}")
                except Exception as e:
                    print(f"Error triggering {ep}: {e}")
        print("Finished triggering submodules.")

    # 2. Read and merge results from all available report files
    all_anomalies_dfs = []
    for ep, config in REPORT_CONFIGS.items():
        report_file_path = os.path.join(config["report_directory"], config["report_filename"])
        df = read_and_standardize_report(report_file_path, config["module_name"])
        if not df.empty:
            all_anomalies_dfs.append(df)

    if not all_anomalies_dfs:
        raise HTTPException(
            status_code=404,
            detail="No anomaly reports found or generated across modules. Ensure sub-APIs run successfully or reports exist."
        )

    # Concatenate all dataframes into one unified dataframe
    merged_df = pd.concat(all_anomalies_dfs, ignore_index=True)

    # 3. Rewrite the unified CSV report
    os.makedirs(UNIFIED_REPORT_DIR, exist_ok=True)
    unified_report_path = os.path.join(UNIFIED_REPORT_DIR, UNIFIED_REPORT_FILENAME)
    merged_df.to_csv(unified_report_path, index=False)
    print(f"Unified anomaly report saved to: {unified_report_path}")

    # 4. Prepare and return the JSON response
    # Convert DataFrame to a list of dictionaries for JSON serialization
    # Ensure numpy NaN values are converted to None for proper JSON
    merged_df_records = merged_df.to_dict(orient="records")

    return {
        "status": "success",
        "total_anomalies": len(merged_df_records),
        "anomalies": merged_df_records,
        "unified_report_path": unified_report_path # Include the path to the unified report in the response
    }


# ------------------ Server Runner ------------------
if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 8989
    print(f"Starting API server at http://{HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)

