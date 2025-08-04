import base64
import datetime
import io
import os
import shutil
import uuid
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import plotly.graph_objects as go
from matplotlib import cm, pyplot as plt # Keeping these imports even if direct SVG doesn't use them, for consistency with original file context.
from pydantic import BaseModel, Field
import regex as re # Keeping this import
import pandas as pd # Keeping this import
from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile, status, Request
from typing import List, Literal, Optional, Dict, Any, Union
import json # Keeping this import
import numpy as np # Import numpy for numerical operations

router = APIRouter()

# --- Pydantic Model for ChartPlottingRequest ---
# In-memory cache for storing chart data
chart_cache: Dict[str, Dict[str, Any]] = {}

# Set up Jinja2 templates.
templates = Jinja2Templates(directory="templates")


# --- Pydantic Models for Data Validation ---

# 3D Data Models
class Series3DData(BaseModel):
    x: List[Any]
    y: List[Any]
    z: List[Any]
    name: str

class Heatmap3DData(BaseModel):
    x: List[Any]
    y: List[Any]
    z: List[List[Any]]
    name: str = "3D Heatmap"

# 2D Data Models
class ChartDataPoint(BaseModel):
    x: List[Any]
    y: List[Any]
    name: str

class PieChartData(BaseModel):
    labels: List[str]
    values: List[float]
    hole: float = 0.4
    hoverinfo: str = "label+percent"

# The main request model is now more comprehensive
class ChartPlottingRequest(BaseModel):
    title: str = "Dynamic Data Visualization"
    description: str = "A powerful, interactive chart generated on-demand."
    chart_type: str = Field(..., description="The type of chart to render (e.g., 'line', 'bar', 'pie', 'scatter', 'heatmap').")
    chart_data: Union[List[ChartDataPoint], PieChartData, List[Series3DData], Heatmap3DData] = Field(..., description="Data payload based on chart type.")
    
    @classmethod
    def __discriminator__(cls, v):
        chart_type = v.get("chart_type")
        if chart_type in ["line", "bar", "stacked_bar", "waterfall", "scatter"]:
            return "chart_data"
        elif chart_type == "pie":
            return "chart_data"
        elif chart_type in ["line3d", "bar3d", "scatter3d"]:
            return "chart_data"
        elif chart_type == "heatmap":
            return "chart_data"
        return None

# --- Helper Function for Chart Generation ---
def generate_plotly_chart_definition(request_data: ChartPlottingRequest) -> Dict[str, Any]:
    """Generates a Plotly chart definition with a futuristic theme."""
    fig = go.Figure()
    
    # A vibrant, futuristic color palette
    color_palette = ['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FF8C00', '#1E90FF', '#FF1493']

    # --- Conditional Logic for All Supported Graph Types (2D & 3D) ---
    chart_type_lower = request_data.chart_type.lower()
    
    if chart_type_lower in ["line", "bar", "stacked_bar", "waterfall", "scatter"]:
        # 2D charts with enhanced styles
        if chart_type_lower == "line":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Scatter(x=series.x, y=series.y, mode='lines+markers', name=series.name,
                    line=dict(color=color_palette[i % len(color_palette)], width=3, shape='spline'),
                    marker=dict(size=10, color='white', line=dict(width=2, color=color_palette[i % len(color_palette)])),
                    hoverinfo='x+y+name'
                ))
        elif chart_type_lower == "bar":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Bar(x=series.x, y=series.y, name=series.name,
                    marker_color=color_palette[i % len(color_palette)],
                    hoverinfo='x+y+name'
                ))
        elif chart_type_lower == "stacked_bar":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Bar(x=series.x, y=series.y, name=series.name,
                    marker_color=color_palette[i % len(color_palette)],
                    hoverinfo='x+y+name'
                ))
            fig.update_layout(barmode='stack')
        elif chart_type_lower == "waterfall":
            series = request_data.chart_data[0]
            fig = go.Figure(go.Waterfall(x=series.x, y=series.y, name=series.name,
                connector=dict(line=dict(color="#555555")),
                increasing=dict(marker=dict(color='#2ECC71')),
                decreasing=dict(marker=dict(color='#E74C3C')),
                totals=dict(marker=dict(color='#3498DB')),
                hoverinfo='x+y+name'
            ))
        elif chart_type_lower == "scatter":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Scatter(x=series.x, y=series.y, mode='markers', name=series.name,
                    marker=dict(size=10, color=color_palette[i % len(color_palette)], line=dict(width=2, color='white')),
                    hoverinfo='x+y+name'
                ))
        
        # Apply 2D layout
        fig.update_layout(
            title_text=f"<b>{request_data.title}</b>",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Roboto Mono, monospace', color='#F0F0F0', size=14),
            xaxis=dict(title_text="X-Axis", showgrid=True, gridcolor='rgba(255,255,255,0.1)', linecolor='rgba(255,255,255,0.2)'),
            yaxis=dict(title_text="Y-Axis", showgrid=True, gridcolor='rgba(255,255,255,0.1)', linecolor='rgba(255,255,255,0.2)'),
            hovermode='closest'
        )

    elif chart_type_lower in ["line3d", "scatter3d", "bar3d", "heatmap"]:
        # 3D charts with enhanced styles
        if chart_type_lower == "line3d":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Scatter3d(x=series.x, y=series.y, z=series.z, mode='lines+markers', name=series.name,
                    line=dict(color=color_palette[i % len(color_palette)], width=4),
                    marker=dict(size=5, symbol='circle', color=color_palette[i % len(color_palette)]),
                    hoverinfo='x+y+z+name'
                ))
        elif chart_type_lower == "scatter3d":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Scatter3d(x=series.x, y=series.y, z=series.z, mode='markers', name=series.name,
                    marker=dict(size=6, symbol='circle', color=color_palette[i % len(color_palette)], opacity=0.9),
                    hoverinfo='x+y+z+name'
                ))
        elif chart_type_lower == "bar3d":
            for i, series in enumerate(request_data.chart_data):
                fig.add_trace(go.Bar3d(x=series.x, y=series.y, z=series.z, name=series.name,
                    marker=dict(color=color_palette[i % len(color_palette)], opacity=0.8),
                    hoverinfo='x+y+z+name'
                ))
        elif chart_type_lower == "heatmap":
            fig.add_trace(go.Surface(
                x=request_data.chart_data.x, y=request_data.chart_data.y, z=request_data.chart_data.z,
                colorscale='Viridis', colorbar_title="Value",
                contours=dict(x=dict(show=True, project=dict(z=True))),
                name=request_data.chart_data.name
            ))

        # Apply 3D layout
        fig.update_layout(
            title_text=f"<b>{request_data.title}</b>",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Roboto Mono, monospace', color='#F0F0F0', size=14),
            scene=dict(
                xaxis={"gridcolor": 'rgba(0, 255, 255, 0.2)', "zerolinecolor": '#00FFFF'},
                yaxis={"gridcolor": 'rgba(255, 0, 255, 0.2)', "zerolinecolor": '#FF00FF'},
                zaxis={"gridcolor": 'rgba(255, 255, 0, 0.2)', "zerolinecolor": '#FFFF00'},
                bgcolor='rgba(0,0,0,0)',
                camera=dict(eye={"x": 1.5, "y": 1.5, "z": 0.8})
            )
        )

    elif chart_type_lower == "pie":
        # Pie chart with enhanced styles
        fig.add_trace(go.Pie(
            labels=request_data.chart_data.labels, values=request_data.chart_data.values,
            name=request_data.title, hole=request_data.chart_data.hole,
            marker_colors=color_palette, textinfo='label+percent',
            insidetextorientation='radial'
        ))
        fig.update_traces(marker=dict(line=dict(color='#FFFFFF', width=2)))
        fig.update_layout(
            title_text=f"<b>{request_data.title}</b>",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Roboto Mono, monospace', color='#F0F0F0', size=14)
        )
    
    else:
        raise ValueError(f"Unsupported chart type: {request_data.chart_type}")
    
    return fig.to_json()


# --- FastAPI Endpoints ---
@router.post("/plot_chart")
async def generate_chart_link(request_data: ChartPlottingRequest = Body(...)):
    """Generates a chart definition, stores it, and returns a unique URL."""
    try:
        chart_definition_dict = json.loads(generate_plotly_chart_definition(request_data))
        chart_definition_dict["description"] = request_data.description
        
        chart_id = str(uuid.uuid4())
        chart_cache[chart_id] = {"data": chart_definition_dict}
        chart_url = f"/mcp/model_management/chart_viewer/{chart_id}"
        return JSONResponse(content={"status": "success", "url": chart_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/chart_viewer/{chart_id}")
async def chart_viewer(request: Request, chart_id: str):
    """Renders the HTML page for a specific chart ID."""
    chart_data_entry = chart_cache.get(chart_id)
    if not chart_data_entry:
        raise HTTPException(status_code=404, detail="Chart not found. The link may have expired.")
    
    chart_definition_json = json.dumps(chart_data_entry["data"])
    
    return templates.TemplateResponse(
        "chart_viewer.html",
        {"request": request, "chart_definition": chart_definition_json}
    )

