import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional
import io
import numpy as np
import os # Added for os.path.join

# Assuming app/utils/file_resolver.py contains the provided code
from app.utils.file_resolver import BASE_DATA_REPORTS_DIR, resolve_input_file_path

# --- FastAPI Router Initialization ---
router = APIRouter()

# Ensure BASE_DATA_REPORTS_DIR exists (important for the dynamic file lookup)
os.makedirs(BASE_DATA_REPORTS_DIR, exist_ok=True)
# Example: Ensure a specific date directory exists for testing
# For "May 15", the directory expected is "May15_Reports"
# os.makedirs(os.path.join(BASE_DATA_REPORTS_DIR, "May15_Reports"), exist_ok=True)
# And you would need to manually place a file like:
# data/Performance_reports_5GR&D_and_wifilab/May15_Reports/ap_clients_5_min_daily_15-05-2025 16_31_16.csv


# --- Pydantic Models for Request Payloads ---

class FilterCondition(BaseModel):
    """Defines a single filter condition for a column."""
    operator: str = Field(..., description="Comparison operator (e.g., 'eq', 'gt', 'lt', 'gte', 'lte', 'between', 'in', 'contains', 'is_null', 'is_not_null').")
    value: Any = Field(None, description="The value for the operator. For 'between', this is the lower bound. For 'in', this must be a list.")
    value2: Optional[Any] = Field(None, description="The second value for operators like 'between' (e.g., upper bound).")

# --- MODIFIED: Pydantic Model for Querying Stored Data with date in payload ---
class StoredDataQueryPayload(BaseModel):
    """Payload for querying stored CSV data by date."""
    date_str: str = Field(..., description="Date for the report (e.g., '2025-05-15', 'May 15').")
    # This 'metric_name' will be specifically "ap clients 5 min daily" for this focus
    metric_name: Literal["ap clients 5 min daily"] = Field(..., description="The specific metric/report to fetch.")
    metrics: List[str] = Field(default_factory=list, description="List of metric names (column names) to retrieve. Required if 'return_type' is 'records'.")
    filters: Optional[Dict[str, FilterCondition]] = Field(default_factory=dict, description="Dictionary of filters to apply (column_name: FilterCondition).")
    aggregate: Optional[Dict[str, List[str]]] = Field(default_factory=dict, description="Dictionary of aggregations to perform (e.g., {'count': ['*'], 'avg': ['rssi']}).")
    return_type: Optional[Literal["records", "count_only", "aggregated_values"]] = Field("records", description="Type of data to return: 'records', 'count_only', 'aggregated_values'.")


# --- METRIC_FILE_MAP for dynamic file resolution (FOCUSED) ---
# This is crucial for resolve_input_file_path to know how to find files
METRIC_FILE_MAP = {
    # This prefix "ap_clients_5_min_daily" matches the example file name.
    # The timestamp part (e.g., 16_31_16) is handled by the `*` in glob within file_resolver.
    "ap clients 5 min daily": {
        "type": "dynamic",
        "prefix": "ap_clients_5_min_daily"
    }
    # Other metrics are removed for focus
}


# --- Helper Functions for Data Processing ---
# (These remain the same as they are generic for DataFrame manipulation)

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names and handles missing/infinite values for a DataFrame.
    This is tailored for the structure of the uploaded client data.
    """
    df.rename(columns={
        'AP Name': 'ap_name', 'Client Mac': 'client_mac', 'Channel': 'channel',
        'Timestamp': 'timestamp', 'Host Name': 'host_name', 'Device Type': 'device_type',
        'Device OS': 'device_os', 'RSSI Strength': 'rssi', 'SNR': 'snr',
        'Client IPv4': 'client_ipv4', 'Client IPv6': 'client_ipv6', 'Site Name': 'site_name',
        'Site Location': 'site_location', 'Site Description': 'site_description',
        'AP Description': 'ap_description', 'Device IPv4': 'device_ipv4',
        'Device IPv6': 'device_ipv6', 'AP Mac': 'ap_mac', 'User Name': 'user_name',
        'SSID': 'ssid', 'Data Uploaded (MB)': 'data_uploaded_mb',
        'Data Downloaded (MB)': 'data_downloaded_mb', 'Interface': 'interface',
        'Radio ID': 'radio_id', 'Uptime': 'uptime', 'Association Time': 'association_time',
        'Disassociation Time': 'disassociation_time', 'Protocol': 'protocol',
        'Created At': 'created_at', 'SAP ID': 'sap_id', 'Group Name': 'group_name',
        'apGroupName': 'ap_group_name', 'BSSID': 'bssid'
    }, inplace=True)

    df.columns = [col.replace(' ', '_').replace('.', '').lower() for col in df.columns]

    numeric_cols = [
        'rssi', 'snr', 'data_uploaded_mb', 'data_downloaded_mb'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(0, inplace=True)
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col].fillna(False, inplace=True)
            else:
                df[col].fillna('unknown', inplace=True)
    return df

def apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, FilterCondition]]) -> pd.DataFrame:
    """Applies a list of filters to a DataFrame."""
    if not filters:
        return df

    filtered_df = df.copy()
    for column, filter_cond in filters.items():
        if column not in filtered_df.columns:
            raise HTTPException(status_code=400, detail=f"Filter column '{column}' not found in data. Available: {list(filtered_df.columns)}")

        col_series = filtered_df[column]

        if filter_cond.operator == "eq":
            filtered_df = filtered_df[col_series == filter_cond.value]
        elif filter_cond.operator == "ne":
            filtered_df = filtered_df[col_series != filter_cond.value]
        elif filter_cond.operator == "gt":
            if not pd.api.types.is_numeric_dtype(col_series):
                raise HTTPException(status_code=400, detail=f"Cannot apply 'gt' filter to non-numeric column '{column}'.")
            filtered_df = filtered_df[col_series > filter_cond.value]
        elif filter_cond.operator == "lt":
            if not pd.api.types.is_numeric_dtype(col_series):
                raise HTTPException(status_code=400, detail=f"Cannot apply 'lt' filter to non-numeric column '{column}'.")
            filtered_df = filtered_df[col_series < filter_cond.value]
        elif filter_cond.operator == "gte":
            if not pd.api.types.is_numeric_dtype(col_series):
                raise HTTPException(status_code=400, detail=f"Cannot apply 'gte' filter to non-numeric column '{column}'.")
            filtered_df = filtered_df[col_series >= filter_cond.value]
        elif filter_cond.operator == "lte":
            if not pd.api.types.is_numeric_dtype(col_series):
                raise HTTPException(status_code=400, detail=f"Cannot apply 'lte' filter to non-numeric column '{column}'.")
            filtered_df = filtered_df[col_series <= filter_cond.value]
        elif filter_cond.operator == "between":
            if not pd.api.types.is_numeric_dtype(col_series):
                raise HTTPException(status_code=400, detail=f"Cannot apply 'between' filter to non-numeric column '{column}'.")
            if filter_cond.value is None or filter_cond.value2 is None:
                raise HTTPException(status_code=400, detail=f"'between' operator requires 'value' and 'value2' for column '{column}'.")
            lower = min(filter_cond.value, filter_cond.value2)
            upper = max(filter_cond.value, filter_cond.value2)
            filtered_df = filtered_df[
                (col_series >= lower) & (col_series <= upper)
            ]
        elif filter_cond.operator == "in":
            if not isinstance(filter_cond.value, list):
                raise HTTPException(status_code=400, detail=f"'in' operator requires 'value' to be a list for column '{column}'.")
            filtered_df = filtered_df[col_series.isin(filter_cond.value)]
        elif filter_cond.operator == "contains":
            str_series = col_series.astype(str).fillna('')
            filtered_df = filtered_df[str_series.str.contains(str(filter_cond.value), na=False)]
        elif filter_cond.operator == "is_null":
            filtered_df = filtered_df[col_series.isnull()]
        elif filter_cond.operator == "is_not_null":
            filtered_df = filtered_df[col_series.notnull()]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported filter operator: '{filter_cond.operator}' for column '{column}'. Supported: eq, ne, gt, lt, gte, lte, between, in, contains, is_null, is_not_null."
            )
    return filtered_df

def perform_aggregation(df: pd.DataFrame, aggregate: Optional[Dict[str, List[str]]]) -> Dict[str, Any]:
    """Performs aggregation on a DataFrame."""
    if not aggregate:
        return {}

    aggregated_results = {}
    for agg_type, cols in aggregate.items():
        if agg_type == "count":
            if "*" in cols:
                aggregated_results["total_filtered_count"] = len(df)
            else:
                for col in cols:
                    if col in df.columns:
                        aggregated_results[f"count_{col}"] = df[col].count()
                    else:
                        print(f"Warning: Count requested for non-existent column '{col}'. Skipping.")
        elif agg_type in ["avg", "sum", "min", "max", "median", "std"]:
            for col in cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if agg_type == "avg": aggregated_results[f"avg_{col}"] = df[col].mean()
                    if agg_type == "sum": aggregated_results[f"sum_{col}"] = df[col].sum()
                    if agg_type == "min": aggregated_results[f"min_{col}"] = df[col].min()
                    if agg_type == "max": aggregated_results[f"max_{col}"] = df[col].max()
                    if agg_type == "median": aggregated_results[f"median_{col}"] = df[col].median()
                    if agg_type == "std": aggregated_results[f"std_{col}"] = df[col].std()
                elif col in df.columns:
                    print(f"Warning: Cannot calculate {agg_type} for non-numeric column '{col}'. Skipping.")
        else:
            print(f"Warning: Unsupported aggregation type: '{agg_type}'. Skipping.")
    return aggregated_results


# --- Helper function to avoid code duplication for processing query results ---
def _process_query_results(
    filtered_df: pd.DataFrame, 
    metrics: List[str], 
    filters: Optional[Dict[str, FilterCondition]], 
    aggregate: Optional[Dict[str, List[str]]], 
    return_type: str
) -> Dict[str, Any]:
    """Internal helper to handle the common logic for returning query results."""
    if return_type == "count_only":
        return {
            "status": "success",
            "filters_applied": filters,
            "total_count_after_filters": len(filtered_df)
        }
    elif return_type == "aggregated_values":
        if not aggregate:
            raise HTTPException(status_code=400, detail="'aggregate' field is required when 'return_type' is 'aggregated_values'.")
        
        aggregated_results = perform_aggregation(filtered_df, aggregate)
        if not aggregated_results and aggregate:
             raise HTTPException(status_code=400, detail="No valid aggregations could be performed with the provided payload. Check column names and types.")
        
        return {
            "status": "success",
            "filters_applied": filters,
            "aggregations_performed": aggregate,
            "data": aggregated_results
        }
    else: # Default return_type is 'records'
        if not metrics:
            raise HTTPException(
                status_code=400,
                detail="'metrics' list is required when 'return_type' is 'records'."
            )

        existing_metrics = [m for m in metrics if m in filtered_df.columns]
        non_existent_metrics = [m for m in metrics if m not in existing_metrics]
        
        if not existing_metrics:
            raise HTTPException(status_code=400, detail=f"No valid metrics found in the data after standardization or filtering. Non-existent requested: {non_existent_metrics}. Available after filters: {list(filtered_df.columns)}")
        
        selected_data = filtered_df[existing_metrics]
        
        return {
            "status": "success",
            "requested_metrics": metrics,
            "returned_metrics_count": len(existing_metrics),
            "total_records": len(selected_data),
            "non_existent_metrics": non_existent_metrics,
            "filters_applied": filters,
            "data": selected_data.to_dict(orient="records")
        }


# --- API Endpoint for querying stored data by date (date in payload) ---
@router.post("/get_info") # Specific endpoint name for this report
async def query_ap_clients_by_date(
    payload: StoredDataQueryPayload
):
    """
    Fetches the 'ap_clients_5_min_daily' CSV report for a given date (from payload),
    and then applies filters and aggregations.
    """
    try:
        # Enforce the specific metric_name for this endpoint
        if payload.metric_name != "ap clients 5 min daily":
            raise HTTPException(
                status_code=400, 
                detail=f"This endpoint is only for 'ap clients 5 min daily' reports. Received: '{payload.metric_name}'"
            )

        # 1. Resolve the file path using the utility function and METRIC_FILE_MAP
        file_path = resolve_input_file_path(payload.date_str, payload.metric_name, METRIC_FILE_MAP)
        
        # 2. Read the file content into a DataFrame
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found at '{file_path}'. Please ensure the file exists and the date/metric_name are correct.")

        df = pd.read_csv(file_path)

        # 3. Standardize the column names and handle missing values
        # The standardize_dataframe function should handle the column names as found in your CSV.
        df = standardize_dataframe(df)

        # 4. Apply filters
        filtered_df = apply_filters(df, payload.filters)

        # 5. Process and return results based on return_type
        return _process_query_results(filtered_df, payload.metrics, payload.filters, payload.aggregate, payload.return_type)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")