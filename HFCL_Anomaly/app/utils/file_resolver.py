# app/utils/file_resolver.py
import os
import glob
from datetime import datetime
from fastapi import HTTPException, status
from typing import Dict, Any

# Base directory where performance reports are stored
# This is the 'data/Performance_reports_5GR&D_and_wifilab/' part of the path
BASE_DATA_REPORTS_DIR = "data/Performance_reports_5GR&D_and_wifilab"

def resolve_input_file_path(date_str: str, metric_name: str, metric_file_map: Dict[str, Dict[str, str]]) -> str:
    """
    Resolves the full path to an input CSV file based on date, metric name,
    and a provided mapping configuration.

    Args:
        date_str: Date string (e.g., 'May 15', '2025-05-15').
        metric_name: User-friendly metric name (e.g., 'ap clients', 'performance anomalies').
        metric_file_map: A dictionary mapping metric names to their file resolution configuration.
                        Each config can be:
                        - {"type": "static", "filename": "some_static_file.csv"}
                        - {"type": "dynamic", "prefix": "some_dynamic_prefix"}

    Returns:
        The full path to the resolved input file.

    Raises:
        HTTPException: If the metric is unknown, date cannot be parsed, or file is not found.
    """
    # FIX: Use metric_name directly to get the config from the map
    config = metric_file_map.get(metric_name)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown metric '{metric_name}'. Supported metrics: {', '.join(metric_file_map.keys())}"
        )

    file_type = config.get("type")

    if file_type == "static":
        filename = config.get("filename")
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Configuration error for static metric '{metric_name}': 'filename' is missing."
            )
        file_path = os.path.join("./", filename) # Assume root for uploaded static files
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Static file for metric '{metric_name}' not found at: '{file_path}'. "
                        "Please ensure 'all_anomalies_with_top5_metrics.csv' is in the project root."
            )
        return file_path

    elif file_type == "dynamic":
        file_prefix = config.get("prefix")
        if not file_prefix:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Configuration error for dynamic metric '{metric_name}': 'prefix' is missing."
            )

        # Parse the date string and determine the date-specific subdirectory and filename date part
        try:
            # Prioritize YYYY-MM-DD or YYYY/MM/DD
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                try:
                    date_obj = datetime.strptime(date_str, '%Y/%m/%d')
                except ValueError:
                    # Handle "Month DD" format (e.g., "May 15"), default year to 2025
                    date_obj = datetime.strptime(date_str + ' 2025', '%B %d %Y')
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse date '{date_str}'. Please use format 'YYYY-MM-DD', 'YYYY/MM/DD', or 'Month DD' (e.g., 'May 15')."
            )

        # Format the date for the date-specific directory name (e.g., "May15_Reports")
        # And for the filename pattern (DD-MM-YYYY)
        date_sub_directory = date_obj.strftime('%b%d_Reports') # e.g., "May15_Reports"
        formatted_date_for_filename = date_obj.strftime('%d-%m-%Y')

        # Construct the full base path to the day's reports directory
        reports_day_dir = os.path.join(BASE_DATA_REPORTS_DIR, date_sub_directory)
        
        # Ensure the date-specific directory exists, as it's part of the expected path
        if not os.path.isdir(reports_day_dir):
             raise HTTPException(
                 status_code=status.HTTP_404_NOT_FOUND,
                 detail=f"Date-specific data directory not found for '{date_str}': '{reports_day_dir}'"
             )

        # Construct the glob pattern to find the file within that directory
        search_pattern = os.path.join(reports_day_dir, f"{file_prefix}_{formatted_date_for_filename}*.csv")
        
        # Use glob to find the file(s) matching the pattern
        found_files = glob.glob(search_pattern)

        if not found_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No file found for metric '{metric_name}' on date '{date_str}'. Searched for pattern: '{search_pattern}'"
            )
        elif len(found_files) > 1:
            # If multiple files match, pick the first one or implement a more specific selection logic
            print(f"Warning: Multiple files found for pattern '{search_pattern}'. Using the first one: {found_files[0]}")
            return found_files[0]
        else:
            return found_files[0]
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid file_type '{file_type}' configured for metric '{metric_name}'."
        )