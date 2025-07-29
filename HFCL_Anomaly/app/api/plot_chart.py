import base64
import datetime
import io
import os
import shutil
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from matplotlib import cm, pyplot as plt # Keeping these imports even if direct SVG doesn't use them, for consistency with original file context.
from pydantic import BaseModel, Field
import regex as re # Keeping this import
import pandas as pd # Keeping this import
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status, Request
from typing import List, Literal, Optional, Dict, Any, Union
import json # Keeping this import
import numpy as np # Import numpy for numerical operations

router = APIRouter()

# --- Pydantic Model for ChartPlottingRequest ---
class ChartPlottingRequest(BaseModel):
    """Request payload for plotting charts using direct SVG generation."""
    x_data: List[Union[float, str]] = Field(..., description="Data points for the X-axis. Can be numerical for quantitative axes or strings for categorical labels.")
    y_data_series: List[List[float]] = Field(..., description="List of Y-axis data series. Each inner list is a series.")
    
    chart_type: Literal["line", "bar", "scatter", "area", "stacked_bar", "horizontal_bar"] = Field("line", description="Type of chart to generate.")
    
    title: str = Field("Chart", description="Title of the chart.")
    x_label: str = Field("X-axis", description="Label for the X-axis.")
    y_label: str = Field("Y-axis", description="Label for the Y-axis.")
    
    width: int = Field(800, description="Width of the SVG chart in pixels.")
    height: int = Field(500, description="Height of the SVG chart in pixels.")
    
    series_labels: Optional[List[str]] = Field(None, description="Optional labels for each data series, matching the order of y_data_series. Used for legend.")
    marker_style: bool = Field(False, description="For line/scatter charts, whether to add markers to data points.")
    line_width: int = Field(2, description="For line/area charts, the width of the lines/borders.")
    fill_opacity: float = Field(0.3, ge=0.0, le=1.0, description="For area charts, the opacity of the fill color.")
    bar_gap_ratio: float = Field(0.2, ge=0.0, le=1.0, description="Ratio of gap between bars to bar width. For grouped bars, ratio of gap between groups.")
    
    x_axis_is_numeric: bool = Field(False, description="If true, x_data will be treated as numerical values for scaling (e.g., for true scatter plots); otherwise, it's treated as categorical labels where points are evenly spaced. Applicable for 'line' and 'scatter'.")


# --- Main Plotting Endpoint ---
@router.post("/plot_chart", response_class=HTMLResponse)
async def plot_chart_html(request: ChartPlottingRequest):
    """
    Generates an SVG chart (line, bar, scatter, area, stacked_bar, horizontal_bar) 
    based on provided data, labels, and dimensions.
    Returns the SVG content embedded in basic HTML.
    """
    if not request.y_data_series or not request.x_data:
        raise HTTPException(status_code=400, detail="Missing x_data or y_data_series.")
    
    # Validate lengths
    for series_index, y_values in enumerate(request.y_data_series):
        if len(y_values) != len(request.x_data):
            raise HTTPException(
                status_code=400, 
                detail=f"Length of y_data_series[{series_index}] ({len(y_values)}) must match length of x_data ({len(request.x_data)})."
            )
    if request.series_labels and len(request.series_labels) != len(request.y_data_series):
        raise HTTPException(status_code=400, detail="Number of series_labels must match y_data_series.")

    chart_type = request.chart_type.lower()
    supported_chart_types = ["line", "bar", "scatter", "area", "stacked_bar", "horizontal_bar"]
    if chart_type not in supported_chart_types:
        raise HTTPException(status_code=400, detail=f"Chart type '{chart_type}' is not supported. Supported types: {', '.join(supported_chart_types)}.")

    # Define SVG dimensions and padding
    svg_width = request.width
    svg_height = request.height
    padding = 60 # Padding for axes and labels

    # --- Determine Axis Scaling ---
    # For horizontal bar charts, swap width/height roles for scaling
    is_horizontal = (chart_type == "horizontal_bar")
    plot_width = svg_width - 2 * padding
    plot_height = svg_height - 2 * padding

    # X-axis scaling
    x_min_val, x_max_val = 0, len(request.x_data) - 1 # Default for categorical x-axis (index-based)
    x_is_numeric_scale = request.x_axis_is_numeric and all(isinstance(val, (int, float)) for val in request.x_data)

    if x_is_numeric_scale:
        x_numeric_data = [float(x) for x in request.x_data]
        if x_numeric_data:
            x_min_val = min(x_numeric_data)
            x_max_val = max(x_numeric_data)
            x_range_buffer = (x_max_val - x_min_val) * 0.1
            if x_range_buffer == 0: # Handle single point or all same x value
                x_range_buffer = 1.0 # Default buffer for single point
            x_min_val -= x_range_buffer
            x_max_val += x_range_buffer
            if x_max_val == x_min_val: x_max_val += 1.0 # Ensure range for single point

    # Y-axis scaling (find min/max across all series)
    all_y_values = [item for sublist in request.y_data_series for item in sublist]
    if not all_y_values:
        y_min_val, y_max_val = 0, 1 # Default if no data
    elif chart_type in ["stacked_bar", "stacked_area"]:
        # Calculate max cumulative sum for stacked charts
        max_cumulative_y = 0
        for i in range(len(request.x_data)):
            current_x_sum = sum(request.y_data_series[series_idx][i] for series_idx in range(len(request.y_data_series)))
            if current_x_sum > max_cumulative_y:
                max_cumulative_y = current_x_sum
        y_min_val = min(0, min(all_y_values)) # Stacked can start from 0 or go negative
        y_max_val = max_cumulative_y
    else:
        y_min_val = min(all_y_values)
        y_max_val = max(all_y_values)

    y_range_buffer = (y_max_val - y_min_val) * 0.1
    if y_range_buffer == 0 and y_max_val == 0: # All zeros
        y_max_scaled = 1.0
        y_min_scaled = -1.0
    elif y_range_buffer == 0: # All identical non-zero values
        y_max_scaled = y_max_val + 1.0
        y_min_scaled = y_min_val - 1.0
    else:
        y_min_scaled = y_min_val - y_range_buffer
        y_max_scaled = y_max_val + y_range_buffer
    

    # Scaling functions to map data coordinates to SVG pixel coordinates
    # For horizontal bar, x_data maps to Y-axis and y_data maps to X-axis
    def scale_x_coord(val, for_axis_label=False): # Maps x_data value/index to SVG x-coordinate
        if is_horizontal: # For horizontal bar, x_data maps to vertical position
            idx = request.x_data.index(val) if not isinstance(val, int) else val
            return padding + plot_height - (idx / (len(request.x_data) - 1 if len(request.x_data) > 1 else 1)) * plot_height # Y-pos for x_data category
        else: # Normal x-axis
            if x_is_numeric_scale and not for_axis_label: # For actual data points (scatter/line)
                return padding + ((val - x_min_val) / (x_max_val - x_min_val)) * plot_width
            else: # Categorical x-axis or for label positioning
                idx = request.x_data.index(val) if not isinstance(val, int) else val
                return padding + (idx / (len(request.x_data) - 1 if len(request.x_data) > 1 else 1)) * plot_width

    def scale_y_coord(val, for_axis_label=False): # Maps y_data value to SVG y-coordinate
        if is_horizontal: # For horizontal bar, y_data maps to horizontal position
            return padding + ((val - y_min_scaled) / (y_max_scaled - y_min_scaled)) * plot_width
        else: # Normal y-axis
            return (svg_height - padding) - ((val - y_min_scaled) / (y_max_scaled - y_min_scaled)) * plot_height

    svg_elements = []
    
    # Define colors for series
    colors = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#00bcd4", "#e91e63", "#66aa00", "#b82912", "#31659a"]

    # Draw Axes Lines
    if is_horizontal:
        svg_elements.append(f'<line x1="{padding}" y1="{svg_height - padding}" x2="{svg_width - padding}" y2="{svg_height - padding}" stroke="black" stroke-width="1"/>') # X-axis (values)
        svg_elements.append(f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{svg_height - padding}" stroke="black" stroke-width="1"/>') # Y-axis (categories)
    else:
        svg_elements.append(f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{svg_height - padding}" stroke="black" stroke-width="1"/>') # Y-axis
        svg_elements.append(f'<line x1="{padding}" y1="{svg_height - padding}" x2="{svg_width - padding}" y2="{svg_height - padding}" stroke="black" stroke-width="1"/>') # X-axis

    # X-axis labels and ticks
    num_x_labels = len(request.x_data)
    for i, label_val in enumerate(request.x_data):
        if is_horizontal: # X-axis is now value axis
            pass # X-axis labels are handled by `y_axis_labels_and_ticks` logic when horizontal
        else:
            x_pos = scale_x_coord(label_val, for_axis_label=True)
            svg_elements.append(f'<text x="{x_pos}" y="{svg_height - padding + 15}" text-anchor="middle" font-size="10">{label_val}</text>')
            svg_elements.append(f'<line x1="{x_pos}" y1="{svg_height - padding}" x2="{x_pos}" y2="{svg_height - padding + 5}" stroke="black" stroke-width="1"/>')
            # Add vertical grid lines
            svg_elements.append(f'<line x1="{x_pos}" y1="{padding}" x2="{x_pos}" y2="{svg_height - padding}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2,2"/>')

    # Y-axis labels and ticks (simplified to 5 ticks)
    num_y_ticks = 5
    for i in range(num_y_ticks):
        y_tick_val = y_min_scaled + (i / (num_y_ticks - 1)) * (y_max_scaled - y_min_scaled) if num_y_ticks > 1 else y_min_scaled
        
        if is_horizontal: # Y-axis is now categorical (x_data labels)
            # This loop handles the numerical value axis (what was Y-axis) for horizontal charts
            x_tick_pos = scale_y_coord(y_tick_val, for_axis_label=True) # Scale y_tick_val to X-pixel for horizontal
            svg_elements.append(f'<text x="{x_tick_pos}" y="{svg_height - padding + 15}" text-anchor="middle" font-size="10">{y_tick_val:.2f}</text>')
            svg_elements.append(f'<line x1="{x_tick_pos}" y1="{svg_height - padding}" x2="{x_tick_pos}" y2="{svg_height - padding + 5}" stroke="black" stroke-width="1"/>')
            # Add vertical grid lines based on X-axis (value axis) ticks
            svg_elements.append(f'<line x1="{x_tick_pos}" y1="{padding}" x2="{x_tick_pos}" y2="{svg_height - padding}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2,2"/>')
            
            # Now, for the actual categorical Y-axis labels
            for idx, label_val in enumerate(request.x_data):
                y_pos_category = padding + plot_height - (idx / (len(request.x_data) - 1 if len(request.x_data) > 1 else 1)) * plot_height
                svg_elements.append(f'<text x="{padding - 10}" y="{y_pos_category + 4}" text-anchor="end" font-size="10">{label_val}</text>')
                svg_elements.append(f'<line x1="{padding}" y1="{y_pos_category}" x2="{padding - 5}" y2="{y_pos_category}" stroke="black" stroke-width="1"/>')
                # Add horizontal grid lines for categorical Y-axis
                svg_elements.append(f'<line x1="{padding}" y1="{y_pos_category}" x2="{svg_width - padding}" y2="{y_pos_category}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2,2"/>')

        else: # Normal vertical Y-axis
            y_pos = scale_y_coord(y_tick_val, for_axis_label=True)
            svg_elements.append(f'<text x="{padding - 10}" y="{y_pos + 4}" text-anchor="end" font-size="10">{y_tick_val:.2f}</text>')
            svg_elements.append(f'<line x1="{padding}" y1="{y_pos}" x2="{padding - 5}" y2="{y_pos}" stroke="black" stroke-width="1"/>')
            # Add horizontal grid lines
            svg_elements.append(f'<line x1="{padding}" y1="{y_pos}" x2="{svg_width - padding}" y2="{y_pos}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2,2"/>')


    # --- Plot Data Series ---
    
    # Calculate zero line position for bar/area charts
    zero_y_pos = scale_y_coord(0) # For vertical charts
    zero_x_pos = scale_y_coord(0) # For horizontal bar chart (y_data maps to x-axis)


    if chart_type in ["line", "scatter", "area"]:
        for i, y_values in enumerate(request.y_data_series):
            current_color = colors[i % len(colors)]
            points = []
            
            # Special handling for area chart path start/end
            if chart_type == "area":
                # Start path from the zero line (or min_scaled) at the first x-point
                x_start_area = scale_x_coord(request.x_data[0])
                points.append(f"{x_start_area},{zero_y_pos}")

            for j, y_val in enumerate(y_values):
                x_val_for_point = request.x_data[j] if x_is_numeric_scale else j
                x_pos = scale_x_coord(x_val_for_point)
                y_pos = scale_y_coord(y_val)
                points.append(f"{x_pos},{y_pos}")

                if chart_type == "scatter" or (chart_type == "line" and request.marker_style):
                    svg_elements.append(f'<circle cx="{x_pos}" cy="{y_pos}" r="4" fill="{current_color}" stroke="none"/>')

            if points:
                if chart_type == "line":
                    path_d = "M " + " L ".join(points)
                    svg_elements.append(f'<path d="{path_d}" stroke="{current_color}" stroke-width="{request.line_width}" fill="none"/>')
                elif chart_type == "area":
                    # Close the path back to the zero line at the last x-point
                    x_end_area = scale_x_coord(request.x_data[-1])
                    points.append(f"{x_end_area},{zero_y_pos}")
                    path_d = "M " + " L ".join(points)
                    svg_elements.append(f'<path d="{path_d}" fill="{current_color}" fill-opacity="{request.fill_opacity}" stroke="{current_color}" stroke-width="{request.line_width}"/>')
    
    elif chart_type in ["bar", "stacked_bar"]:
        num_categories = len(request.x_data)
        total_series = len(request.y_data_series)
        
        # Calculate bar dimensions
        category_total_width = plot_width / num_categories
        bar_gap = category_total_width * request.bar_gap_ratio
        
        if chart_type == "bar": # Grouped bars
            bar_width = (category_total_width - bar_gap) / total_series
        else: # Stacked bars
            bar_width = category_total_width - bar_gap # Each stack occupies this width

        # Ensure bar width is positive
        bar_width = max(1, bar_width) 

        # Store current accumulated heights for stacked bars
        if chart_type == "stacked_bar":
            current_y_stack = [0.0] * num_categories # For positive stacks
            current_y_stack_neg = [0.0] * num_categories # For negative stacks

        for j in range(num_categories): # Iterate through each x_data point (category)
            x_pos_category_center = scale_x_coord(j) # Center of the current x-category
            
            if chart_type == "bar": # Grouped bars
                # Calculate the starting X position for the first bar in this group
                group_start_x = x_pos_category_center - (bar_width * total_series + bar_gap * (total_series - 1)) / 2
                
                for i, y_values in enumerate(request.y_data_series): # Iterate through each series
                    y_val = y_values[j]
                    current_color = colors[i % len(colors)]
                    
                    x_rect_start = group_start_x + (i * bar_width) + (i * bar_gap / 2) # Adjust for individual bar position
                    
                    y_pos_top = scale_y_coord(y_val)
                    bar_y_start = min(y_pos_top, zero_y_pos)
                    bar_height = abs(y_pos_top - zero_y_pos)

                    if bar_height < 0: bar_height = 0
                    svg_elements.append(f'<rect x="{x_rect_start}" y="{bar_y_start}" width="{bar_width}" height="{bar_height}" fill="{current_color}"/>')
            
            elif chart_type == "stacked_bar":
                x_rect_start = x_pos_category_center - (bar_width / 2)
                for i, y_values in enumerate(request.y_data_series):
                    y_val = y_values[j]
                    current_color = colors[i % len(colors)]

                    if y_val >= 0:
                        y_pos_top = scale_y_coord(current_y_stack[j] + y_val)
                        y_pos_bottom = scale_y_coord(current_y_stack[j])
                        bar_y_start = y_pos_top
                        bar_height = y_pos_bottom - y_pos_top
                        current_y_stack[j] += y_val
                    else: # Negative stack
                        y_pos_top = scale_y_coord(current_y_stack_neg[j])
                        y_pos_bottom = scale_y_coord(current_y_stack_neg[j] + y_val)
                        bar_y_start = y_pos_top
                        bar_height = y_pos_bottom - y_pos_top
                        current_y_stack_neg[j] += y_val

                    if bar_height < 0: bar_height = 0 # Ensure non-negative height
                    svg_elements.append(f'<rect x="{x_rect_start}" y="{bar_y_start}" width="{bar_width}" height="{bar_height}" fill="{current_color}"/>')

    elif chart_type == "horizontal_bar":
        num_categories = len(request.x_data)
        total_series = len(request.y_data_series)
        
        category_total_height = plot_height / num_categories
        bar_gap = category_total_height * request.bar_gap_ratio

        bar_height_per_series = (category_total_height - bar_gap) / total_series
        bar_height_per_series = max(1, bar_height_per_series) # Ensure positive height

        for j in range(num_categories): # Iterate through each x_data point (category)
            y_pos_category_center = scale_y_coord(request.x_data[j]) # Center of the current y-category (which is x_data)
            
            # Calculate the starting Y position for the first bar in this group
            group_start_y = y_pos_category_center - (bar_height_per_series * total_series + bar_gap * (total_series - 1)) / 2

            for i, y_values in enumerate(request.y_data_series): # Iterate through each series
                y_val = y_values[j]
                current_color = colors[i % len(colors)]
                
                y_rect_start = group_start_y + (i * bar_height_per_series) + (i * bar_gap / 2)
                
                x_pos_end = scale_y_coord(y_val) # X-position of the bar's end (value axis)
                bar_x_start = min(x_pos_end, zero_x_pos)
                bar_width = abs(x_pos_end - zero_x_pos)

                if bar_width < 0: bar_width = 0
                svg_elements.append(f'<rect x="{bar_x_start}" y="{y_rect_start}" width="{bar_width}" height="{bar_height_per_series}" fill="{current_color}"/>')


    # Axis labels
    if is_horizontal:
        svg_elements.append(f'<text x="{svg_width / 2}" y="{svg_height - 10}" text-anchor="middle" font-size="14">{request.y_label}</text>') # X-axis (values)
        svg_elements.append(f'<text x="{15}" y="{svg_height / 2}" text-anchor="middle" transform="rotate(-90 {15},{svg_height / 2})" font-size="14">{request.x_label}</text>') # Y-axis (categories)
    else:
        svg_elements.append(f'<text x="{svg_width / 2}" y="{svg_height - 10}" text-anchor="middle" font-size="14">{request.x_label}</text>')
        svg_elements.append(f'<text x="{15}" y="{svg_height / 2}" text-anchor="middle" transform="rotate(-90 {15},{svg_height / 2})" font-size="14">{request.y_label}</text>')

    # Title
    svg_elements.append(f'<text x="{svg_width / 2}" y="30" text-anchor="middle" font-size="18" font-weight="bold">{request.title}</text>')

    # Basic Legend (optional, for multiple series)
    if request.series_labels and len(request.y_data_series) > 0:
        legend_start_x = svg_width - 150
        legend_start_y = 50
        for i, label in enumerate(request.series_labels):
            color = colors[i % len(colors)]
            svg_elements.append(f'<rect x="{legend_start_x}" y="{legend_start_y + i * 20}" width="15" height="15" fill="{color}"/>')
            svg_elements.append(f'<text x="{legend_start_x + 20}" y="{legend_start_y + i * 20 + 12}" font-size="12">{label}</text>')


    # Combine all SVG elements into a full SVG string
    svg_content = f"""
    <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="{svg_width}" height="{svg_height}" fill="white"/>
        {" ".join(svg_elements)}
    </svg>
    """

    # Final HTML with embedded SVG
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{request.title}</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; }}
            svg {{ border: 1px solid #ccc; background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        {svg_content}
    </body>
    </html>
    """
    return HTMLResponse(content=html)