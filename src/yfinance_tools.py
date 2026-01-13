"""
Yahoo Finance Tools for MCP Server
Provides access to stock data, financial statements, options, and more via yfinance
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# Define enums for financial statement types
class FinancialType(str, Enum):
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"


class HolderType(str, Enum):
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


class RecommendationType(str, Enum):
    recommendations = "recommendations"
    upgrades_downgrades = "upgrades_downgrades"


def _validate_ticker(ticker: str) -> tuple[yf.Ticker, str | None]:
    """Validate ticker and return Ticker object or error message."""
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            return company, f"Company ticker {ticker} not found."
    except Exception as e:
        return company, f"Error validating ticker {ticker}: {e}"
    return company, None


@dataclass
class BaseYFinanceTool:
    """Base class for Yahoo Finance tools."""
    
    config: Dict[str, Any]
    
    def __post_init__(self):
        self.name = self.config.get("name", "yfinance_tool")
        self.description = self.config.get("description", "Yahoo Finance tool")
        self.api_base = "local"
        self.input_schema = self.config.get("input_schema", {})


class GetHistoricalStockPricesTool(BaseYFinanceTool):
    """Get historical stock prices for a given ticker symbol."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.__post_init__()
        self.name = config.get("name", "get_historical_stock_prices")
        self.description = config.get("description", """Get historical stock prices for a given ticker symbol from yahoo finance. 
Includes: Date, Open, High, Low, Close, Volume, Adj Close.

Args:
    ticker: The ticker symbol (e.g. "AAPL")
    period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max (default: "1mo")
    interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (default: "1d")
""")
        self.input_schema = {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"},
                "period": {"type": "string", "description": "Time period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)", "default": "1mo"},
                "interval": {"type": "string", "description": "Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)", "default": "1d"}
            },
            "required": ["ticker"]
        }
    
    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ticker = arguments.get("ticker")
        period = arguments.get("period", "1mo")
        interval = arguments.get("interval", "1d")
        
        if not ticker:
            return {"error": "ticker is required"}
        
        company, error = _validate_ticker(ticker)
        if error:
            return {"content": error}
        
        try:
            hist_data = company.history(period=period, interval=interval)
            hist_data = hist_data.reset_index(names="Date")
            result = hist_data.to_json(orient="records", date_format="iso")
            return {"content": result}
        except Exception as e:
            logger.error(f"Error getting historical prices for {ticker}: {e}")
            return {"content": f"Error getting historical stock prices for {ticker}: {e}"}


class GetStockInfoTool(BaseYFinanceTool):
    """Get stock information for a given ticker symbol."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.__post_init__()
        self.name = config.get("name", "get_stock_info")
        self.description = config.get("description", """Get stock information for a given ticker symbol from yahoo finance.
Includes: Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, 
Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics.

Args:
    ticker: The ticker symbol (e.g. "AAPL")
""")
        self.input_schema = {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"}
            },
            "required": ["ticker"]
        }
    
    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ticker = arguments.get("ticker")
        
        if not ticker:
            return {"error": "ticker is required"}
        
        company, error = _validate_ticker(ticker)
        if error:
            return {"content": error}
        
        try:
            info = company.info
            return {"content": json.dumps(info)}
        except Exception as e:
            logger.error(f"Error getting stock info for {ticker}: {e}")
            return {"content": f"Error getting stock information for {ticker}: {e}"}


class GetYahooFinanceNewsTool(BaseYFinanceTool):
    """Get news for a given ticker symbol."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.__post_init__()
        self.name = config.get("name", "get_yahoo_finance_news")
        self.description = config.get("description", """Get news for a given ticker symbol from yahoo finance.

Args:
    ticker: The ticker symbol (e.g. "AAPL")
""")
        self.input_schema = {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"}
            },
            "required": ["ticker"]
        }
    
    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ticker = arguments.get("ticker")
        
        if not ticker:
            return {"error": "ticker is required"}
        
        company, error = _validate_ticker(ticker)
        if error:
            return {"content": error}
        
        try:
            news = company.news
            news_list = []
            for article in news:
                if article.get("content", {}).get("contentType", "") == "STORY":
                    title = article.get("content", {}).get("title", "")
                    summary = article.get("content", {}).get("summary", "")
                    description = article.get("content", {}).get("description", "")
                    url = article.get("content", {}).get("canonicalUrl", {}).get("url", "")
                    news_list.append(
                        f"Title: {title}\nSummary: {summary}\nDescription: {description}\nURL: {url}"
                    )
            
            if not news_list:
                return {"content": f"No news found for company with ticker {ticker}."}
            
            return {"content": "\n\n".join(news_list)}
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {e}")
            return {"content": f"Error getting news for {ticker}: {e}"}


class GetStockActionsTool(BaseYFinanceTool):
    """Get stock dividends and stock splits for a given ticker symbol."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.__post_init__()
        self.name = config.get("name", "get_stock_actions")
        self.description = config.get("description", """Get stock dividends and stock splits for a given ticker symbol from yahoo finance.

Args:
    ticker: The ticker symbol (e.g. "AAPL")
""")
        self.input_schema = {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"}
            },
            "required": ["ticker"]
        }
    
    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ticker = arguments.get("ticker")
        
        if not ticker:
            return {"error": "ticker is required"}
        
        try:
            company = yf.Ticker(ticker)
            actions_df = company.actions
            actions_df = actions_df.reset_index(names="Date")
            return {"content": actions_df.to_json(orient="records", date_format="iso")}
        except Exception as e:
            logger.error(f"Error getting stock actions for {ticker}: {e}")
            return {"content": f"Error getting stock actions for {ticker}: {e}"}


class GetFinancialStatementTool(BaseYFinanceTool):
    """Get financial statement for a given ticker symbol."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.__post_init__()
        self.name = config.get("name", "get_financial_statement")
        self.description = config.get("description", """Get financial statement for a given ticker symbol from yahoo finance.

Financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, 
quarterly_balance_sheet, cashflow, quarterly_cashflow.

Args:
    ticker: The ticker symbol (e.g. "AAPL")
    financial_type: Type of financial statement
""")
        self.input_schema = {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"},
                "financial_type": {
                    "type": "string",
                    "description": "Type of financial statement",
                    "enum": ["income_stmt", "quarterly_income_stmt", "balance_sheet", 
                             "quarterly_balance_sheet", "cashflow", "quarterly_cashflow"]
                }
            },
            "required": ["ticker", "financial_type"]
        }
    
    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ticker = arguments.get("ticker")
        financial_type = arguments.get("financial_type")
        
        if not ticker:
            return {"error": "ticker is required"}
        if not financial_type:
            return {"error": "financial_type is required"}
        
        company, error = _validate_ticker(ticker)
        if error:
            return {"content": error}
        
        try:
            if financial_type == FinancialType.income_stmt.value:
                financial_statement = company.income_stmt
            elif financial_type == FinancialType.quarterly_income_stmt.value:
                financial_statement = company.quarterly_income_stmt
            elif financial_type == FinancialType.balance_sheet.value:
                financial_statement = company.balance_sheet
            elif financial_type == FinancialType.quarterly_balance_sheet.value:
                financial_statement = company.quarterly_balance_sheet
            elif financial_type == FinancialType.cashflow.value:
                financial_statement = company.cashflow
            elif financial_type == FinancialType.quarterly_cashflow.value:
                financial_statement = company.quarterly_cashflow
            else:
                return {"content": f"Error: invalid financial type {financial_type}. Valid types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow."}
            
            # Convert to list of date-keyed objects
            result = []
            for column in financial_statement.columns:
                if isinstance(column, pd.Timestamp):
                    date_str = column.strftime("%Y-%m-%d")
                else:
                    date_str = str(column)
                
                date_obj = {"date": date_str}
                for index, value in financial_statement[column].items():
                    date_obj[index] = None if pd.isna(value) else value
                
                result.append(date_obj)
            
            return {"content": json.dumps(result)}
        except Exception as e:
            logger.error(f"Error getting financial statement for {ticker}: {e}")
            return {"content": f"Error getting financial statement for {ticker}: {e}"}


class GetHolderInfoTool(BaseYFinanceTool):
    """Get holder information for a given ticker symbol."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.__post_init__()
        self.name = config.get("name", "get_holder_info")
        self.description = config.get("description", """Get holder information for a given ticker symbol from yahoo finance.

Holder types: major_holders, institutional_holders, mutualfund_holders, 
insider_transactions, insider_purchases, insider_roster_holders.

Args:
    ticker: The ticker symbol (e.g. "AAPL")
    holder_type: Type of holder information
""")
        self.input_schema = {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"},
                "holder_type": {
                    "type": "string",
                    "description": "Type of holder information",
                    "enum": ["major_holders", "institutional_holders", "mutualfund_holders",
                             "insider_transactions", "insider_purchases", "insider_roster_holders"]
                }
            },
            "required": ["ticker", "holder_type"]
        }
    
    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ticker = arguments.get("ticker")
        holder_type = arguments.get("holder_type")
        
        if not ticker:
            return {"error": "ticker is required"}
        if not holder_type:
            return {"error": "holder_type is required"}
        
        company, error = _validate_ticker(ticker)
        if error:
            return {"content": error}
        
        try:
            if holder_type == HolderType.major_holders.value:
                return {"content": company.major_holders.reset_index(names="metric").to_json(orient="records")}
            elif holder_type == HolderType.institutional_holders.value:
                return {"content": company.institutional_holders.to_json(orient="records")}
            elif holder_type == HolderType.mutualfund_holders.value:
                return {"content": company.mutualfund_holders.to_json(orient="records", date_format="iso")}
            elif holder_type == HolderType.insider_transactions.value:
                return {"content": company.insider_transactions.to_json(orient="records", date_format="iso")}
            elif holder_type == HolderType.insider_purchases.value:
                return {"content": company.insider_purchases.to_json(orient="records", date_format="iso")}
            elif holder_type == HolderType.insider_roster_holders.value:
                return {"content": company.insider_roster_holders.to_json(orient="records", date_format="iso")}
            else:
                return {"content": f"Error: invalid holder type {holder_type}. Valid types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders."}
        except Exception as e:
            logger.error(f"Error getting holder info for {ticker}: {e}")
            return {"content": f"Error getting holder info for {ticker}: {e}"}


class GetOptionExpirationDatesTool(BaseYFinanceTool):
    """Fetch the available options expiration dates for a given ticker symbol."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.__post_init__()
        self.name = config.get("name", "get_option_expiration_dates")
        self.description = config.get("description", """Fetch the available options expiration dates for a given ticker symbol.

Args:
    ticker: The ticker symbol (e.g. "AAPL")
""")
        self.input_schema = {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"}
            },
            "required": ["ticker"]
        }
    
    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ticker = arguments.get("ticker")
        
        if not ticker:
            return {"error": "ticker is required"}
        
        company, error = _validate_ticker(ticker)
        if error:
            return {"content": error}
        
        try:
            return {"content": json.dumps(company.options)}
        except Exception as e:
            logger.error(f"Error getting option expiration dates for {ticker}: {e}")
            return {"content": f"Error getting option expiration dates for {ticker}: {e}"}


class GetOptionChainTool(BaseYFinanceTool):
    """Fetch the option chain for a given ticker symbol, expiration date, and option type."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.__post_init__()
        self.name = config.get("name", "get_option_chain")
        self.description = config.get("description", """Fetch the option chain for a given ticker symbol, expiration date, and option type.

Args:
    ticker: The ticker symbol (e.g. "AAPL")
    expiration_date: The expiration date (format: 'YYYY-MM-DD')
    option_type: The type of option ('calls' or 'puts')
""")
        self.input_schema = {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"},
                "expiration_date": {"type": "string", "description": "Expiration date (YYYY-MM-DD)"},
                "option_type": {"type": "string", "description": "Option type", "enum": ["calls", "puts"]}
            },
            "required": ["ticker", "expiration_date", "option_type"]
        }
    
    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ticker = arguments.get("ticker")
        expiration_date = arguments.get("expiration_date")
        option_type = arguments.get("option_type")
        
        if not ticker:
            return {"error": "ticker is required"}
        if not expiration_date:
            return {"error": "expiration_date is required"}
        if not option_type:
            return {"error": "option_type is required"}
        
        company, error = _validate_ticker(ticker)
        if error:
            return {"content": error}
        
        try:
            # Check if the expiration date is valid
            if expiration_date not in company.options:
                return {"content": f"Error: No options available for the date {expiration_date}. Use get_option_expiration_dates to get available dates."}
            
            # Check if the option type is valid
            if option_type not in ["calls", "puts"]:
                return {"content": "Error: Invalid option type. Please use 'calls' or 'puts'."}
            
            # Get the option chain
            option_chain = company.option_chain(expiration_date)
            if option_type == "calls":
                return {"content": option_chain.calls.to_json(orient="records", date_format="iso")}
            else:
                return {"content": option_chain.puts.to_json(orient="records", date_format="iso")}
        except Exception as e:
            logger.error(f"Error getting option chain for {ticker}: {e}")
            return {"content": f"Error getting option chain for {ticker}: {e}"}


class GetRecommendationsTool(BaseYFinanceTool):
    """Get recommendations or upgrades/downgrades for a given ticker symbol."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.__post_init__()
        self.name = config.get("name", "get_recommendations")
        self.description = config.get("description", """Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance.

Recommendation types: recommendations, upgrades_downgrades

Args:
    ticker: The ticker symbol (e.g. "AAPL")
    recommendation_type: Type of recommendation
    months_back: Number of months back for upgrades/downgrades (default: 12)
""")
        self.input_schema = {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"},
                "recommendation_type": {
                    "type": "string",
                    "description": "Type of recommendation",
                    "enum": ["recommendations", "upgrades_downgrades"]
                },
                "months_back": {"type": "integer", "description": "Months back for upgrades/downgrades", "default": 12}
            },
            "required": ["ticker", "recommendation_type"]
        }
    
    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ticker = arguments.get("ticker")
        recommendation_type = arguments.get("recommendation_type")
        months_back = arguments.get("months_back", 12)
        
        if not ticker:
            return {"error": "ticker is required"}
        if not recommendation_type:
            return {"error": "recommendation_type is required"}
        
        company, error = _validate_ticker(ticker)
        if error:
            return {"content": error}
        
        try:
            if recommendation_type == RecommendationType.recommendations.value:
                return {"content": company.recommendations.to_json(orient="records")}
            elif recommendation_type == RecommendationType.upgrades_downgrades.value:
                # Get the upgrades/downgrades based on the cutoff date
                upgrades_downgrades = company.upgrades_downgrades.reset_index()
                cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
                upgrades_downgrades = upgrades_downgrades[
                    upgrades_downgrades["GradeDate"] >= cutoff_date
                ]
                upgrades_downgrades = upgrades_downgrades.sort_values("GradeDate", ascending=False)
                # Get the first occurrence (most recent) for each firm
                latest_by_firm = upgrades_downgrades.drop_duplicates(subset=["Firm"])
                return {"content": latest_by_firm.to_json(orient="records", date_format="iso")}
            else:
                return {"content": f"Error: invalid recommendation type {recommendation_type}. Valid types: recommendations, upgrades_downgrades."}
        except Exception as e:
            logger.error(f"Error getting recommendations for {ticker}: {e}")
            return {"content": f"Error getting recommendations for {ticker}: {e}"}
