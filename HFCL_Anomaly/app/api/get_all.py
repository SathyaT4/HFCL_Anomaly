import os
import numpy as np
import pandas as pd
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from app.utils.file_resolver import BASE_DATA_REPORTS_DIR

router = APIRouter()

# Ensure directories exist
os.makedirs("./reports/performance_anomalies/", exist_ok=True)
os.makedirs("./reports/interference_detection/", exist_ok=True)
os.makedirs("./reports/connection_drops/", exist_ok=True)
os.makedirs("./reports/channel_overlap/", exist_ok=True)
os.makedirs("./models/", exist_ok=True)
os.makedirs(BASE_DATA_REPORTS_DIR, exist_ok=True)

# Configuration for reports
REPORT_CONFIGS = {
    "/performance/detect_anomalies": {
        "module_name": "Performance",
        "report_filename": "performance_anomalies_report.csv",
        "report_directory": "./reports/performance_anomalies/"
    },
    "/optimization/detect_channel_overlap": {
        "module_name": "ChannelOverlap",
        "report_filename": "channel_optimization_report.csv",
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


def load_report_as_json(file_path: str, module_name: str) -> list[dict]:
    """
    Reads a CSV report and returns it as a JSON-compliant list of dicts.
    Handles NaN and Infinity values for JSON serialization.
    """
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"Warning: {file_path} is empty. Skipping.")
            return []

        # Add source module
        df['source_module'] = module_name

        # Replace problematic values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Convert Inf -> NaN
        df.fillna(value=np.nan, inplace=True)  # Ensure NaN for all empty cells

        # Convert NaN to None for JSON
        df = df.astype(object).where(pd.notnull(df), None)

        return df.to_dict(orient="records")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


@router.get("/anomalies/all_reports", response_model=Dict[str, Any])
async def get_all_reports_grouped():
    """
    Reads all anomaly report CSVs and returns them as JSON grouped by module.
    """
    all_reports_data = {}

    for ep, config in REPORT_CONFIGS.items():
        module_name = config["module_name"]
        report_file_path = os.path.join(config["report_directory"], config["report_filename"])

        # Load CSV as JSON list
        report_data = load_report_as_json(report_file_path, module_name)
        all_reports_data[module_name] = report_data

    # If no reports are found
    if not any(all_reports_data.values()):
        raise HTTPException(
            status_code=404,
            detail="No anomaly reports found. Please ensure individual anomaly detection APIs have generated reports."
        )

    return all_reports_data
