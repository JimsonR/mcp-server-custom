"""
Chart Plotting Tools for MCP Server
Generates structured chart data for frontend visualization without overloading LLM context.
Supports: line, candlestick, bar, area, multi_line charts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    """Supported chart types."""
    line = "line"
    candlestick = "candlestick"
    bar = "bar"
    area = "area"
    multi_line = "multi_line"


# Maximum data points to return (prevents response bloat)
MAX_DATA_POINTS = 60


def _validate_ticker(ticker: str) -> tuple[yf.Ticker, str | None]:
    """Validate ticker and return Ticker object or error message."""
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return company, f"Company ticker {ticker} not found."
    except Exception as e:
        return company, f"Error validating ticker {ticker}: {e}"
    return company, None


def _calculate_summary_stats(df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """Calculate summary statistics for the chart data."""
    if df.empty:
        return {}
    
    close_col = "Close" if "Close" in df.columns else "close"
    if close_col not in df.columns:
        return {}
    
    close_prices = df[close_col].dropna()
    if close_prices.empty:
        return {}
    
    first_price = close_prices.iloc[0]
    last_price = close_prices.iloc[-1]
    change_pct = ((last_price - first_price) / first_price) * 100 if first_price != 0 else 0
    
    return {
        "ticker": ticker,
        "start_price": round(first_price, 2),
        "end_price": round(last_price, 2),
        "high": round(close_prices.max(), 2),
        "low": round(close_prices.min(), 2),
        "change_percent": round(change_pct, 2),
        "data_points": len(close_prices),
        "trend": "up" if change_pct > 0 else "down" if change_pct < 0 else "flat"
    }


def _generate_text_summary(stats: Dict[str, Any], chart_type: str) -> str:
    """Generate a concise text summary for LLM context (keeps context small)."""
    if not stats:
        return "Chart generated but no summary statistics available."
    
    trend_emoji = "📈" if stats.get("trend") == "up" else "📉" if stats.get("trend") == "down" else "➡️"
    change_sign = "+" if stats.get("change_percent", 0) >= 0 else ""
    
    return (
        f"{trend_emoji} {chart_type.upper()} Chart: {stats.get('ticker', 'N/A')} | "
        f"Price: ${stats.get('start_price', 0)} → ${stats.get('end_price', 0)} "
        f"({change_sign}{stats.get('change_percent', 0)}%) | "
        f"High: ${stats.get('high', 0)}, Low: ${stats.get('low', 0)} | "
        f"{stats.get('data_points', 0)} data points"
    )


@dataclass
class BaseChartTool:
    """Base class for chart tools."""
    
    config: Dict[str, Any]
    
    def __post_init__(self):
        self.name = self.config.get("name", "chart_tool")
        self.description = self.config.get("description", "Chart tool")
        self.api_base = "local"
        self.input_schema = self.config.get("input_schema", {})


class PlotChartTool(BaseChartTool):
    """
    Generate chart visualization data for frontend rendering.
    Returns structured data optimized for Plotly/Recharts with minimal LLM context impact.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.__post_init__()
        self.name = config.get("name", "plot_chart")
        self.description = config.get("description", """Generate a chart visualization for stock data. 
Returns structured data that the frontend can render as an interactive chart.

Supported chart types:
- line: Simple price line over time
- candlestick: OHLC candlestick chart (best for price action analysis)
- bar: Volume or metric comparison bars
- area: Filled area chart for cumulative metrics
- multi_line: Compare multiple tickers on same chart

Args:
    ticker: Stock ticker symbol (e.g. "AAPL"). For multi_line, comma-separated (e.g. "AAPL,MSFT,GOOGL")
    chart_type: Type of chart to generate (line, candlestick, bar, area, multi_line)
    period: Time period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max). Default: 1mo
    interval: Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo). Default: 1d
    metric: For bar charts - which metric to plot (volume, close, open, high, low). Default: close
""")
        self.input_schema = {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol. For multi_line charts, use comma-separated values (e.g. 'AAPL,MSFT,GOOGL')"
                },
                "chart_type": {
                    "type": "string",
                    "description": "Type of chart to generate",
                    "enum": ["line", "candlestick", "bar", "area", "multi_line"],
                    "default": "line"
                },
                "period": {
                    "type": "string",
                    "description": "Time period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)",
                    "default": "1mo"
                },
                "interval": {
                    "type": "string",
                    "description": "Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)",
                    "default": "1d"
                },
                "metric": {
                    "type": "string",
                    "description": "Metric to plot for bar charts (volume, close, open, high, low)",
                    "enum": ["volume", "close", "open", "high", "low"],
                    "default": "close"
                }
            },
            "required": ["ticker", "chart_type"]
        }
    
    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chart generation request."""
        ticker_input = arguments.get("ticker", "").strip()
        chart_type = arguments.get("chart_type", "line")
        period = arguments.get("period", "1mo")
        interval = arguments.get("interval", "1d")
        metric = arguments.get("metric", "close")
        
        if not ticker_input:
            return {"error": "ticker is required"}
        
        # Validate chart type
        try:
            chart_type_enum = ChartType(chart_type)
        except ValueError:
            valid_types = [t.value for t in ChartType]
            return {"error": f"Invalid chart_type '{chart_type}'. Valid types: {valid_types}"}
        
        try:
            if chart_type_enum == ChartType.multi_line:
                return self._generate_multi_line_chart(ticker_input, period, interval)
            else:
                return self._generate_single_chart(ticker_input, chart_type_enum, period, interval, metric)
        except Exception as e:
            logger.error(f"Error generating chart for {ticker_input}: {e}")
            return {"error": f"Error generating chart: {str(e)}"}
    
    def _generate_single_chart(
        self, 
        ticker: str, 
        chart_type: ChartType, 
        period: str, 
        interval: str,
        metric: str
    ) -> Dict[str, Any]:
        """Generate chart data for a single ticker."""
        company, error = _validate_ticker(ticker)
        if error:
            return {"content": error}
        
        try:
            hist_data = company.history(period=period, interval=interval)
            if hist_data.empty:
                return {"content": f"No historical data available for {ticker}"}
            
            # Limit data points to prevent large responses
            if len(hist_data) > MAX_DATA_POINTS:
                hist_data = hist_data.tail(MAX_DATA_POINTS)
            
            # Reset index to get date as column
            hist_data = hist_data.reset_index()
            
            # Format dates for JSON serialization
            if "Date" in hist_data.columns:
                hist_data["Date"] = hist_data["Date"].dt.strftime("%Y-%m-%d")
            elif "Datetime" in hist_data.columns:
                hist_data["Date"] = hist_data["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
                hist_data = hist_data.drop(columns=["Datetime"])
            
            # Calculate summary stats
            stats = _calculate_summary_stats(hist_data, ticker)
            text_summary = _generate_text_summary(stats, chart_type.value)
            
            # Build chart data based on type
            chart_data = self._build_chart_data(hist_data, chart_type, ticker, metric)
            
            # Build visualization response
            visualization = {
                "type": chart_type.value,
                "library": "plotly",  # Recommended library for frontend
                "config": {
                    "title": f"{ticker} - {chart_type.value.title()} Chart ({period})",
                    "ticker": ticker,
                    "period": period,
                    "interval": interval
                },
                "data": chart_data,
                "layout": self._get_layout_config(chart_type, ticker, metric),
                "summary": stats
            }
            
            return {
                "content": text_summary,  # Concise text for LLM context
                "visualization": visualization  # Structured data for frontend rendering
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return {"error": f"Error fetching stock data: {str(e)}"}
    
    def _generate_multi_line_chart(
        self, 
        tickers_input: str, 
        period: str, 
        interval: str
    ) -> Dict[str, Any]:
        """Generate multi-line comparison chart for multiple tickers."""
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        
        if len(tickers) < 2:
            return {"error": "multi_line chart requires at least 2 comma-separated tickers"}
        
        if len(tickers) > 5:
            return {"error": "multi_line chart supports maximum 5 tickers for readability"}
        
        all_series = []
        summaries = []
        
        for ticker in tickers:
            company, error = _validate_ticker(ticker)
            if error:
                logger.warning(f"Skipping invalid ticker {ticker}: {error}")
                continue
            
            try:
                hist_data = company.history(period=period, interval=interval)
                if hist_data.empty:
                    continue
                
                # Limit data points
                if len(hist_data) > MAX_DATA_POINTS:
                    hist_data = hist_data.tail(MAX_DATA_POINTS)
                
                hist_data = hist_data.reset_index()
                
                # Format dates
                if "Date" in hist_data.columns:
                    dates = hist_data["Date"].dt.strftime("%Y-%m-%d").tolist()
                elif "Datetime" in hist_data.columns:
                    dates = hist_data["Datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
                else:
                    dates = list(range(len(hist_data)))
                
                close_prices = hist_data["Close"].round(2).tolist()
                
                all_series.append({
                    "name": ticker,
                    "x": dates,
                    "y": close_prices,
                    "type": "scatter",
                    "mode": "lines"
                })
                
                stats = _calculate_summary_stats(hist_data, ticker)
                summaries.append(stats)
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                continue
        
        if not all_series:
            return {"error": "Could not fetch data for any of the provided tickers"}
        
        # Generate text summary
        summary_parts = []
        for s in summaries:
            if s:
                trend_emoji = "📈" if s.get("trend") == "up" else "📉" if s.get("trend") == "down" else "➡️"
                summary_parts.append(f"{s.get('ticker', 'N/A')}: {trend_emoji}{s.get('change_percent', 0):+.1f}%")
        
        text_summary = f"📊 Multi-Line Comparison: {' | '.join(summary_parts)}"
        
        visualization = {
            "type": "multi_line",
            "library": "plotly",
            "config": {
                "title": f"Stock Comparison: {', '.join(tickers)} ({period})",
                "tickers": tickers,
                "period": period,
                "interval": interval
            },
            "data": all_series,
            "layout": {
                "xaxis": {"title": "Date", "type": "category"},
                "yaxis": {"title": "Price ($)"},
                "showlegend": True,
                "legend": {"orientation": "h", "y": -0.2}
            },
            "summary": summaries
        }
        
        return {
            "content": text_summary,
            "visualization": visualization
        }
    
    def _build_chart_data(
        self, 
        df: pd.DataFrame, 
        chart_type: ChartType, 
        ticker: str,
        metric: str
    ) -> List[Dict[str, Any]]:
        """Build chart data array based on chart type."""
        dates = df["Date"].tolist()
        
        if chart_type == ChartType.candlestick:
            # Candlestick requires OHLC data
            return [{
                "type": "candlestick",
                "name": ticker,
                "x": dates,
                "open": df["Open"].round(2).tolist(),
                "high": df["High"].round(2).tolist(),
                "low": df["Low"].round(2).tolist(),
                "close": df["Close"].round(2).tolist(),
                "increasing": {"line": {"color": "#26a69a"}},
                "decreasing": {"line": {"color": "#ef5350"}}
            }]
        
        elif chart_type == ChartType.line:
            return [{
                "type": "scatter",
                "mode": "lines",
                "name": ticker,
                "x": dates,
                "y": df["Close"].round(2).tolist(),
                "line": {"color": "#2196F3", "width": 2}
            }]
        
        elif chart_type == ChartType.area:
            return [{
                "type": "scatter",
                "mode": "lines",
                "name": ticker,
                "x": dates,
                "y": df["Close"].round(2).tolist(),
                "fill": "tozeroy",
                "fillcolor": "rgba(33, 150, 243, 0.3)",
                "line": {"color": "#2196F3", "width": 2}
            }]
        
        elif chart_type == ChartType.bar:
            # Bar chart uses specified metric
            metric_col = metric.capitalize() if metric.lower() != "volume" else "Volume"
            if metric_col not in df.columns:
                metric_col = "Close"
            
            values = df[metric_col].round(2).tolist()
            
            return [{
                "type": "bar",
                "name": f"{ticker} {metric.capitalize()}",
                "x": dates,
                "y": values,
                "marker": {"color": "#4CAF50"}
            }]
        
        # Default to line
        return [{
            "type": "scatter",
            "mode": "lines",
            "name": ticker,
            "x": dates,
            "y": df["Close"].round(2).tolist()
        }]
    
    def _get_layout_config(self, chart_type: ChartType, ticker: str, metric: str) -> Dict[str, Any]:
        """Get Plotly layout configuration for the chart type."""
        base_layout = {
            "xaxis": {"title": "Date", "type": "category"},
            "autosize": True,
            "margin": {"l": 50, "r": 50, "t": 50, "b": 50}
        }
        
        if chart_type == ChartType.candlestick:
            base_layout.update({
                "yaxis": {"title": "Price ($)"},
                "xaxis": {"title": "Date", "rangeslider": {"visible": False}, "type": "category"}
            })
        elif chart_type == ChartType.bar and metric.lower() == "volume":
            base_layout["yaxis"] = {"title": "Volume"}
        else:
            base_layout["yaxis"] = {"title": "Price ($)"}
        
        return base_layout
