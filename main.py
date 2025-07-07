#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
                                Image, PageBreak, KeepTogether)
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus.flowables import Flowable
import io
import warnings
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import csv
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum
from collections import defaultdict
import math
import argparse
import re
from contextlib import redirect_stdout
import html

warnings.filterwarnings('ignore')
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Use default style if seaborn is not available

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio analysis"""
    volatility: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float
    annualized_return: float
    alpha: float
    information_ratio: float
    calmar_ratio: float
    sortino_ratio: float

class RiskRating(Enum):
    """Morningstar-style risk ratings"""
    LOW = "Low"
    BELOW_AVERAGE = "Below Average"
    AVERAGE = "Average"
    ABOVE_AVERAGE = "Above Average"
    HIGH = "High"

class QualityRating(Enum):
    """Quality ratings for stocks"""
    EXEMPLARY = "Exemplary"
    STANDARD = "Standard"
    POOR = "Poor"

class ValuationRating(Enum):
    """Valuation assessment ratings"""
    UNDERVALUED = "Undervalued"
    FAIRLY_VALUED = "Fairly Valued"
    SLIGHTLY_OVERVALUED = "Slightly Overvalued"
    OVERVALUED = "Overvalued"
    INSUFFICIENT_DATA = "Insufficient Data"

@dataclass
class StockProfile:
    """Comprehensive stock profile similar to Morningstar"""
    symbol: str
    company_name: str
    sector: str
    industry: str
    market_cap: float
    market_cap_category: str
    
    # Fundamental metrics
    pe_ratio: Optional[float]
    peg_ratio: Optional[float]
    price_to_book: Optional[float]
    price_to_sales: Optional[float]
    debt_to_equity: Optional[float]
    roe: Optional[float]
    roa: Optional[float]
    profit_margin: Optional[float]
    revenue_growth: Optional[float]
    earnings_growth: Optional[float]
    
    # Additional fundamental metrics
    current_ratio: Optional[float]
    quick_ratio: Optional[float]
    gross_margin: Optional[float]
    operating_margin: Optional[float]
    free_cash_flow_yield: Optional[float]
    book_value_per_share: Optional[float]
    tangible_book_value: Optional[float]
    
    # Dividend metrics
    dividend_yield: float
    dividend_growth_rate: Optional[float]
    payout_ratio: Optional[float]
    dividend_sustainability_score: Optional[float]
    
    # Risk metrics
    beta: float
    volatility: float
    risk_rating: RiskRating
    quality_rating: QualityRating
    max_drawdown: float
    var_95: float
    
    # Valuation
    fair_value_estimate: Optional[float]
    margin_of_safety: Optional[float]
    valuation_rating: ValuationRating
    price_to_fair_value: Optional[float]
    
    # Portfolio contribution
    portfolio_weight: float
    contribution_to_return: float
    contribution_to_risk: float
    diversification_benefit: float
    position_size_recommendation: str
    
    # ESG and sustainability
    esg_score: Optional[float]
    sustainability_rank: Optional[str]
    governance_score: Optional[float]
    
    # Performance metrics
    one_year_return: Optional[float]
    three_year_return: Optional[float]
    five_year_return: Optional[float]
    ytd_return: Optional[float]
    
    # Analyst insights
    investment_thesis: str
    risk_factors: List[str]
    opportunities: List[str]
    competitive_advantages: List[str]
    recommendations: List[str]
    
    # Financial strength indicators
    financial_strength_score: Optional[float]
    debt_coverage_ratio: Optional[float]
    interest_coverage_ratio: Optional[float]
    altman_z_score: Optional[float]

class ProgressBar(Flowable):
    """Custom progress bar for PDF reports"""
    def __init__(self, width, height, value, max_value, color=colors.green):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.value = value
        self.max_value = max_value
        self.color = color
        
    def draw(self):
        canvas = self.canv
        canvas.setFillColor(colors.lightgrey)
        canvas.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        
        if self.max_value > 0:
            progress_width = (self.value / self.max_value) * self.width
            canvas.setFillColor(self.color)
            canvas.rect(0, 0, progress_width, self.height, fill=1, stroke=0)
        
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(1)
        canvas.rect(0, 0, self.width, self.height, fill=0, stroke=1)

class EnhancedPortfolioAnalyzer:
    """Enhanced Portfolio Analyzer with comprehensive features"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.df = None
        self.holdings = {}
        self.current_prices = {}
        self.stock_info = {}
        self.historical_data = {}
        self.benchmark_data = {}
        
        # Enhanced color palette
        self.colors = {
            'primary': '#2E86AB',    # Blue
            'secondary': '#A23B72',  # Purple
            'accent': '#F18F01',     # Orange
            'success': '#28A745',    # Green
            'warning': '#FFC107',    # Yellow
            'danger': '#DC3545',     # Red
            'background': '#F8F9FA',
            'text': '#2C3E50'
        }
        
    def load_data(self) -> None:
        """Load and clean the CSV data with enhanced error handling for multi-line fields"""
        try:
            logger.info("Loading portfolio data...")
            
            # First, try standard pandas approach with robust settings
            try:
                self.df = pd.read_csv(
                    self.csv_file,
                    quoting=csv.QUOTE_ALL,
                    skipinitialspace=True,
                    on_bad_lines='warn',
                    engine='python'  # More robust for malformed data
                )
                logger.info(f"Successfully loaded CSV with {len(self.df)} rows using standard method")
                
            except Exception as csv_error:
                logger.warning(f"Standard CSV loading failed: {csv_error}")
                logger.info("Attempting custom CSV parsing for multi-line descriptions...")
                
                # Custom parsing for problematic CSV
                self.df = self._parse_csv_with_multiline_handling()
                logger.info(f"Successfully loaded CSV with {len(self.df)} rows using custom parser")
            
            # Clean and process data
            original_count = len(self.df)
            
            # Clean multi-line descriptions
            if 'Description' in self.df.columns:
                self.df['Description'] = self.df['Description'].astype(str).str.replace(r'\n+', ' ', regex=True)
                self.df['Description'] = self.df['Description'].str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # Parse dates with multiple format support
            date_columns = ['Activity Date', 'Process Date', 'Settle Date']
            for col in date_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
            # Clean amount column - handle various formats
            if 'Amount' in self.df.columns:
                # Handle parentheses notation for negative amounts BEFORE removing characters
                amount_series = self.df['Amount'].astype(str)
                negative_mask = amount_series.str.contains(r'\(.*\)', na=False)
                
                # Clean the amount string
                cleaned_amounts = amount_series.str.replace(r'[\$,()\s]', '', regex=True)
                self.df['Amount'] = pd.to_numeric(cleaned_amounts, errors='coerce').fillna(0)
                
                # Apply negative sign for parentheses notation
                self.df.loc[negative_mask, 'Amount'] = -self.df.loc[negative_mask, 'Amount'].abs()
            
            # Clean price and quantity
            if 'Price' in self.df.columns:
                self.df['Price'] = pd.to_numeric(
                    self.df['Price'].astype(str).str.replace(r'[\$,\s]', '', regex=True), 
                    errors='coerce'
                ).fillna(0)
            
            if 'Quantity' in self.df.columns:
                self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce').fillna(0)
            
            # Remove rows with invalid activity dates
            if 'Activity Date' in self.df.columns:
                self.df = self.df.dropna(subset=['Activity Date'])
            
            # Data quality report
            final_count = len(self.df)
            skipped_count = original_count - final_count
            
            logger.info(f"Data loading complete:")
            logger.info(f"  - Loaded: {final_count} valid transactions")
            if skipped_count > 0:
                logger.info(f"  - Skipped: {skipped_count} rows with invalid data")
            
            # Log column information
            logger.info(f"  - Columns: {list(self.df.columns)}")
            
            if final_count == 0:
                raise ValueError("No valid transactions found in the CSV file")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _parse_csv_with_multiline_handling(self) -> pd.DataFrame:
        """Custom CSV parser for files with multi-line descriptions"""
        rows = []
        current_row = []
        in_quotes = False
        quote_char = None
        
        with open(self.csv_file, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()
        
        # Find the header line
        lines = content.split('\n')
        header_line = None
        header_index = 0
        
        for i, line in enumerate(lines):
            if 'Activity Date' in line or 'Symbol' in line:
                header_line = line
                header_index = i
                break
        
        if header_line is None:
            raise ValueError("Could not find header line in CSV")
        
        # Parse header
        headers = [h.strip('"').strip() for h in header_line.split(',')]
        
        # Process data lines
        data_lines = lines[header_index + 1:]
        
        for line_num, line in enumerate(data_lines, start=header_index + 2):
            if not line.strip():
                continue
                
            # Simple parsing - split by comma but be careful with quoted fields
            fields = []
            current_field = ""
            in_quotes = False
            
            i = 0
            while i < len(line):
                char = line[i]
                
                if char == '"' and not in_quotes:
                    in_quotes = True
                elif char == '"' and in_quotes:
                    # Check if this is an escaped quote
                    if i + 1 < len(line) and line[i + 1] == '"':
                        current_field += '"'
                        i += 1  # Skip the next quote
                    else:
                        in_quotes = False
                elif char == ',' and not in_quotes:
                    fields.append(current_field.strip())
                    current_field = ""
                else:
                    current_field += char
                    
                i += 1
            
            # Add the last field
            fields.append(current_field.strip())
            
            # Pad or trim fields to match header length
            while len(fields) < len(headers):
                fields.append("")
            fields = fields[:len(headers)]
            
            if len(fields) == len(headers):
                rows.append(fields)
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)
        return df
    
    def calculate_holdings(self) -> None:
        """Calculate current holdings from transaction data"""
        try:
            logger.info("Calculating holdings from transaction data...")
            
            if self.df is None:
                raise ValueError("No data loaded. Call load_data() first.")
            
            # Filter for buy/sell transactions
            buy_sell_df = self.df[
                (self.df['Trans Code'].isin(['Buy', 'Sell', 'Dividend Reinvestment'])) &
                (self.df['Instrument'].notna()) &
                (self.df['Instrument'] != '') &
                (self.df['Quantity'].notna()) &
                (self.df['Price'].notna())
            ].copy()
            
            logger.info(f"Processing {len(buy_sell_df)} buy/sell transactions")
            
            # Calculate holdings
            for _, row in buy_sell_df.iterrows():
                symbol = str(row['Instrument']).strip().upper()
                if not symbol or symbol == 'NAN':
                    continue
                    
                action = row['Trans Code']
                quantity = float(row['Quantity']) if pd.notna(row['Quantity']) else 0
                price = float(row['Price']) if pd.notna(row['Price']) else 0
                
                if symbol not in self.holdings:
                    self.holdings[symbol] = {
                        'quantity': 0,
                        'total_cost': 0,
                        'transactions': []
                    }
                
                # Process transaction
                if action in ['Buy', 'Dividend Reinvestment']:
                    self.holdings[symbol]['quantity'] += quantity
                    self.holdings[symbol]['total_cost'] += quantity * price
                elif action == 'Sell':
                    self.holdings[symbol]['quantity'] -= quantity
                    # Reduce total cost proportionally
                    if self.holdings[symbol]['quantity'] > 0:
                        cost_per_share = self.holdings[symbol]['total_cost'] / (self.holdings[symbol]['quantity'] + quantity)
                        self.holdings[symbol]['total_cost'] -= quantity * cost_per_share
                    else:
                        self.holdings[symbol]['total_cost'] = 0
                
                # Store transaction for reference
                self.holdings[symbol]['transactions'].append({
                    'date': row['Activity Date'],
                    'action': action,
                    'quantity': quantity,
                    'price': price
                })
            
            # Calculate average cost and remove zero holdings
            symbols_to_remove = []
            for symbol, holding in self.holdings.items():
                if holding['quantity'] <= 0.001:  # Remove very small positions
                    symbols_to_remove.append(symbol)
                else:
                    holding['avg_cost'] = holding['total_cost'] / holding['quantity'] if holding['quantity'] > 0 else 0
            
            for symbol in symbols_to_remove:
                del self.holdings[symbol]
            
            logger.info(f"Holdings calculation complete: {len(self.holdings)} active positions")
            
        except Exception as e:
            logger.error(f"Error calculating holdings: {e}")
            raise
    
    def calculate_advanced_fundamental_metrics(self, symbol: str, info: Dict, hist_data: pd.DataFrame) -> Dict:
        """Calculate advanced fundamental metrics for a stock"""
        try:
            metrics = {}
            
            # Basic valuation metrics
            metrics['pe_ratio'] = info.get('trailingPE')
            metrics['forward_pe'] = info.get('forwardPE')
            metrics['peg_ratio'] = info.get('pegRatio')
            metrics['price_to_book'] = info.get('priceToBook')
            metrics['price_to_sales'] = info.get('priceToSalesTrailing12Months')
            metrics['enterprise_value'] = info.get('enterpriseValue')
            metrics['ev_to_revenue'] = info.get('enterpriseToRevenue')
            metrics['ev_to_ebitda'] = info.get('enterpriseToEbitda')
            
            # Profitability metrics
            metrics['profit_margin'] = info.get('profitMargins')
            metrics['operating_margin'] = info.get('operatingMargins')
            metrics['gross_margin'] = info.get('grossMargins')
            metrics['return_on_equity'] = info.get('returnOnEquity')
            metrics['return_on_assets'] = info.get('returnOnAssets')
            
            # Growth metrics (from info if available)
            metrics['revenue_growth'] = info.get('revenueGrowth')
            metrics['earnings_growth'] = info.get('earningsGrowth')
            metrics['revenue_growth_quarterly'] = info.get('revenueQuarterlyGrowth')
            metrics['earnings_growth_quarterly'] = info.get('earningsQuarterlyGrowth')
            
            # Financial health metrics
            metrics['debt_to_equity'] = info.get('debtToEquity')
            metrics['current_ratio'] = info.get('currentRatio')
            metrics['quick_ratio'] = info.get('quickRatio')
            metrics['total_cash'] = info.get('totalCash')
            metrics['total_debt'] = info.get('totalDebt')
            metrics['free_cash_flow'] = info.get('freeCashflow')
            metrics['operating_cash_flow'] = info.get('operatingCashflow')
            
            # Per-share metrics
            metrics['book_value_per_share'] = info.get('bookValue')
            metrics['cash_per_share'] = info.get('totalCashPerShare')
            metrics['revenue_per_share'] = info.get('revenuePerShare')
            
            # Dividend metrics
            metrics['dividend_yield'] = info.get('dividendYield', 0) or 0
            metrics['dividend_rate'] = info.get('dividendRate')
            metrics['payout_ratio'] = info.get('payoutRatio')
            metrics['five_year_avg_dividend_yield'] = info.get('fiveYearAvgDividendYield')
            
            # Calculate additional metrics from historical data if available
            if not hist_data.empty:
                metrics['volatility'] = hist_data['Close'].pct_change().std() * np.sqrt(252)
                metrics['max_drawdown'] = self._calculate_max_drawdown(hist_data['Close'])
                
                # Calculate dividend growth if we have dividend data
                metrics['dividend_growth_rate'] = self._calculate_dividend_growth(hist_data)
                
                # Calculate performance metrics
                if len(hist_data) >= 252:  # At least 1 year of data
                    metrics['one_year_return'] = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-252] - 1)
                if len(hist_data) >= 756:  # At least 3 years of data
                    annual_return = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-756] - 1)
                    metrics['three_year_return'] = annual_return / 3
                
                # YTD return
                current_year = datetime.now().year
                ytd_data = hist_data[hist_data.index.year == current_year]
                if len(ytd_data) > 1:
                    metrics['ytd_return'] = (ytd_data['Close'].iloc[-1] / ytd_data['Close'].iloc[0] - 1)
            
            # Calculate composite metrics
            metrics['fair_value_estimate'] = self._estimate_fair_value(info, metrics)
            metrics['margin_of_safety'] = self._calculate_margin_of_safety(
                info.get('currentPrice', 0), 
                metrics.get('fair_value_estimate')
            )
            
            # Financial strength score
            metrics['financial_strength_score'] = self._calculate_financial_strength_score(metrics)
            
            # Altman Z-Score for bankruptcy risk
            metrics['altman_z_score'] = self._calculate_altman_z_score(info, metrics)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating advanced metrics for {symbol}: {e}")
            return {}
    
    def _calculate_dividend_growth(self, hist_data: pd.DataFrame) -> Optional[float]:
        """Calculate dividend growth rate from historical data"""
        try:
            # This is a simplified calculation - in practice you'd want dividend history
            # For now, return None as we don't have dividend-specific historical data
            return None
        except:
            return None
    
    def _estimate_fair_value(self, info: Dict, metrics: Dict) -> Optional[float]:
        """Estimate fair value using multiple valuation methods"""
        try:
            current_price = info.get('currentPrice', 0)
            if not current_price:
                return None
            
            fair_values = []
            
            # P/E based valuation
            pe_ratio = metrics.get('pe_ratio')
            if pe_ratio and pe_ratio > 0:
                # Compare to sector average P/E (simplified)
                sector_pe = self._get_industry_pe_multiple(info.get('sector', ''))
                if sector_pe:
                    eps = current_price / pe_ratio if pe_ratio else 0
                    pe_fair_value = eps * sector_pe
                    fair_values.append(pe_fair_value)
            
            # P/B based valuation
            pb_ratio = metrics.get('price_to_book')
            book_value = metrics.get('book_value_per_share')
            if pb_ratio and book_value and pb_ratio > 0:
                # Assume fair P/B is 1.5 for growth stocks, 1.0 for value stocks
                fair_pb = 1.2  # Conservative estimate
                pb_fair_value = book_value * fair_pb
                fair_values.append(pb_fair_value)
            
            # DCF approximation using free cash flow
            free_cash_flow = info.get('freeCashflow')
            shares_outstanding = info.get('sharesOutstanding')
            if free_cash_flow and shares_outstanding:
                fcf_per_share = free_cash_flow / shares_outstanding
                # Simple DCF with 8% discount rate and 3% growth
                discount_rate = 0.08
                growth_rate = 0.03
                dcf_fair_value = fcf_per_share * (1 + growth_rate) / (discount_rate - growth_rate)
                if dcf_fair_value > 0:
                    fair_values.append(dcf_fair_value)
            
            # Return average of available valuations
            if fair_values:
                return sum(fair_values) / len(fair_values)
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error estimating fair value: {e}")
            return None
    
    def _get_industry_pe_multiple(self, sector: str) -> float:
        """Get typical P/E multiple for a sector"""
        sector_pe_multiples = {
            'Technology': 25.0,
            'Healthcare': 20.0,
            'Financial Services': 12.0,
            'Consumer Cyclical': 18.0,
            'Consumer Defensive': 16.0,
            'Industrial': 18.0,
            'Energy': 15.0,
            'Utilities': 14.0,
            'Real Estate': 16.0,
            'Materials': 15.0,
            'Communication Services': 22.0
        }
        return sector_pe_multiples.get(sector, 18.0)  # Default P/E
    
    def _calculate_margin_of_safety(self, current_price: float, fair_value: Optional[float]) -> Optional[float]:
        """Calculate margin of safety percentage"""
        if fair_value and current_price and fair_value > 0:
            return (fair_value - current_price) / fair_value
        return None
    
    def _calculate_financial_strength_score(self, metrics: Dict) -> Optional[float]:
        """Calculate a composite financial strength score (0-100)"""
        try:
            score = 0
            max_score = 0
            
            # Profitability (40% weight)
            roe = metrics.get('return_on_equity')
            if roe is not None:
                max_score += 40
                if roe > 0.15:  # Excellent
                    score += 40
                elif roe > 0.10:  # Good
                    score += 30
                elif roe > 0.05:  # Average
                    score += 20
                elif roe > 0:  # Below average
                    score += 10
            
            # Financial leverage (25% weight)
            debt_to_equity = metrics.get('debt_to_equity')
            if debt_to_equity is not None:
                max_score += 25
                if debt_to_equity < 0.3:  # Low debt
                    score += 25
                elif debt_to_equity < 0.6:  # Moderate debt
                    score += 20
                elif debt_to_equity < 1.0:  # High debt
                    score += 10
                elif debt_to_equity < 2.0:  # Very high debt
                    score += 5
            
            # Liquidity (20% weight)
            current_ratio = metrics.get('current_ratio')
            if current_ratio is not None:
                max_score += 20
                if current_ratio > 2.0:  # Excellent liquidity
                    score += 20
                elif current_ratio > 1.5:  # Good liquidity
                    score += 15
                elif current_ratio > 1.2:  # Adequate liquidity
                    score += 10
                elif current_ratio > 1.0:  # Minimal liquidity
                    score += 5
            
            # Profit margins (15% weight)
            profit_margin = metrics.get('profit_margin')
            if profit_margin is not None:
                max_score += 15
                if profit_margin > 0.20:  # Excellent margins
                    score += 15
                elif profit_margin > 0.10:  # Good margins
                    score += 12
                elif profit_margin > 0.05:  # Average margins
                    score += 8
                elif profit_margin > 0:  # Low margins
                    score += 4
            
            if max_score > 0:
                return (score / max_score) * 100
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error calculating financial strength score: {e}")
            return None
    
    def _calculate_altman_z_score(self, info: Dict, metrics: Dict) -> Optional[float]:
        """Calculate Altman Z-Score for bankruptcy prediction"""
        try:
            # Altman Z-Score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E where:
            # A = Working Capital / Total Assets
            # B = Retained Earnings / Total Assets  
            # C = EBIT / Total Assets
            # D = Market Value of Equity / Total Liabilities
            # E = Sales / Total Assets
            
            total_assets = info.get('totalAssets')
            if not total_assets:
                return None
            
            # A: Working Capital / Total Assets
            current_assets = info.get('totalCurrentAssets', 0)
            current_liabilities = info.get('totalCurrentLiabilities', 0)
            working_capital = current_assets - current_liabilities
            a = working_capital / total_assets
            
            # B: Retained Earnings / Total Assets
            retained_earnings = info.get('retainedEarnings', 0)
            b = retained_earnings / total_assets
            
            # C: EBIT / Total Assets
            ebit = info.get('ebitda', 0)  # Approximation
            c = ebit / total_assets
            
            # D: Market Value of Equity / Total Liabilities
            market_cap = info.get('marketCap', 0)
            total_liabilities = info.get('totalLiab', 0)
            if total_liabilities > 0:
                d = market_cap / total_liabilities
            else:
                d = 0
            
            # E: Sales / Total Assets
            revenue = info.get('totalRevenue', 0)
            e = revenue / total_assets
            
            # Calculate Z-Score
            z_score = 1.2*a + 1.4*b + 3.3*c + 0.6*d + 1.0*e
            
            return z_score
            
        except Exception as e:
            logger.debug(f"Error calculating Altman Z-Score: {e}")
            return None
    
    def _assign_risk_rating(self, volatility: float, beta: float, financial_strength_score: Optional[float], altman_z_score: Optional[float]) -> RiskRating:
        """Assign risk rating based on multiple factors"""
        try:
            risk_score = 0
            
            # Volatility component (0-40 points)
            if volatility < 0.15:  # Low volatility
                risk_score += 10
            elif volatility < 0.25:  # Medium volatility
                risk_score += 20
            elif volatility < 0.35:  # High volatility
                risk_score += 30
            else:  # Very high volatility
                risk_score += 40
            
            # Beta component (0-30 points)
            if beta < 0.8:  # Low beta
                risk_score += 5
            elif beta < 1.2:  # Average beta
                risk_score += 15
            elif beta < 1.5:  # High beta
                risk_score += 25
            else:  # Very high beta
                risk_score += 30
            
            # Financial strength component (0-20 points, inverted)
            if financial_strength_score is not None:
                if financial_strength_score > 80:  # Strong financials = low risk
                    risk_score += 0
                elif financial_strength_score > 60:  # Good financials
                    risk_score += 5
                elif financial_strength_score > 40:  # Average financials
                    risk_score += 10
                elif financial_strength_score > 20:  # Weak financials
                    risk_score += 15
                else:  # Very weak financials = high risk
                    risk_score += 20
            
            # Altman Z-Score component (0-10 points)
            if altman_z_score is not None:
                if altman_z_score > 3.0:  # Safe zone
                    risk_score += 0
                elif altman_z_score > 1.8:  # Gray zone
                    risk_score += 5
                else:  # Distress zone
                    risk_score += 10
            
            # Convert to rating
            if risk_score <= 20:
                return RiskRating.LOW
            elif risk_score <= 40:
                return RiskRating.BELOW_AVERAGE
            elif risk_score <= 60:
                return RiskRating.AVERAGE
            elif risk_score <= 80:
                return RiskRating.ABOVE_AVERAGE
            else:
                return RiskRating.HIGH
                
        except Exception as e:
            logger.debug(f"Error assigning risk rating: {e}")
            return RiskRating.AVERAGE
    
    def _assign_quality_rating(self, metrics: Dict) -> QualityRating:
        """Assign quality rating based on fundamental metrics"""
        try:
            quality_score = 0
            max_score = 0
            
            # Profitability metrics (40% weight)
            roe = metrics.get('return_on_equity')
            if roe is not None:
                max_score += 40
                if roe > 0.15:
                    quality_score += 40
                elif roe > 0.10:
                    quality_score += 30
                elif roe > 0.05:
                    quality_score += 20
                elif roe > 0:
                    quality_score += 10
            
            # Profit margins (30% weight)
            profit_margin = metrics.get('profit_margin')
            if profit_margin is not None:
                max_score += 30
                if profit_margin > 0.15:
                    quality_score += 30
                elif profit_margin > 0.10:
                    quality_score += 25
                elif profit_margin > 0.05:
                    quality_score += 15
                elif profit_margin > 0:
                    quality_score += 10
            
            # Financial strength (30% weight)
            debt_to_equity = metrics.get('debt_to_equity')
            current_ratio = metrics.get('current_ratio')
            
            if debt_to_equity is not None:
                max_score += 15
                if debt_to_equity < 0.3:
                    quality_score += 15
                elif debt_to_equity < 0.6:
                    quality_score += 10
                elif debt_to_equity < 1.0:
                    quality_score += 5
            
            if current_ratio is not None:
                max_score += 15
                if current_ratio > 2.0:
                    quality_score += 15
                elif current_ratio > 1.5:
                    quality_score += 10
                elif current_ratio > 1.2:
                    quality_score += 5
            
            # Calculate percentage
            if max_score > 0:
                quality_percentage = (quality_score / max_score) * 100
                if quality_percentage >= 80:
                    return QualityRating.EXEMPLARY
                elif quality_percentage >= 60:
                    return QualityRating.STANDARD
                else:
                    return QualityRating.POOR
            else:
                return QualityRating.STANDARD
                
        except Exception as e:
            logger.debug(f"Error assigning quality rating: {e}")
            return QualityRating.STANDARD
    
    def _assign_valuation_rating(self, margin_of_safety: Optional[float]) -> ValuationRating:
        """Assign valuation rating based on margin of safety"""
        if margin_of_safety is None:
            return ValuationRating.INSUFFICIENT_DATA
        
        if margin_of_safety > 0.25:  # 25% or more below fair value
            return ValuationRating.UNDERVALUED
        elif margin_of_safety > 0.10:  # 10-25% below fair value
            return ValuationRating.FAIRLY_VALUED
        elif margin_of_safety > -0.10:  # Within 10% of fair value
            return ValuationRating.FAIRLY_VALUED
        elif margin_of_safety > -0.25:  # 10-25% above fair value
            return ValuationRating.SLIGHTLY_OVERVALUED
        else:  # More than 25% above fair value
            return ValuationRating.OVERVALUED
    
    def _calculate_dividend_sustainability_score(self, payout_ratio: Optional[float], debt_to_equity: Optional[float], 
                                               free_cash_flow: Optional[float], dividend_rate: Optional[float]) -> Optional[float]:
        """Calculate dividend sustainability score (0-100)"""
        try:
            if not dividend_rate or dividend_rate <= 0:
                return None
            
            score = 0
            max_score = 0
            
            # Payout ratio (40% weight)
            if payout_ratio is not None:
                max_score += 40
                if payout_ratio < 0.4:  # Very sustainable
                    score += 40
                elif payout_ratio < 0.6:  # Sustainable
                    score += 30
                elif payout_ratio < 0.8:  # Moderate risk
                    score += 20
                elif payout_ratio < 1.0:  # High risk
                    score += 10
                # Payout ratio > 100% = 0 points
            
            # Financial leverage (35% weight)
            if debt_to_equity is not None:
                max_score += 35
                if debt_to_equity < 0.3:  # Low debt supports dividends
                    score += 35
                elif debt_to_equity < 0.6:  # Moderate debt
                    score += 25
                elif debt_to_equity < 1.0:  # High debt
                    score += 15
                elif debt_to_equity < 2.0:  # Very high debt
                    score += 5
            
            # Free cash flow coverage (25% weight)
            if free_cash_flow is not None and free_cash_flow > 0:
                max_score += 25
                # This is simplified - ideally we'd calculate FCF per share vs dividend per share
                score += 20  # Assume decent coverage if FCF is positive
            
            if max_score > 0:
                return (score / max_score) * 100
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error calculating dividend sustainability score: {e}")
            return None
    
    def get_market_data(self) -> None:
        """Fetch current market data for all holdings with enhanced data collection"""
        logger.info("Fetching comprehensive market data...")
        
        if not self.holdings:
            logger.warning("No holdings found. Run calculate_holdings() first.")
            return
        
        symbols = list(self.holdings.keys())
        logger.info(f"Fetching data for {len(symbols)} symbols: {', '.join(symbols)}")
        
        def fetch_comprehensive_data(symbol: str) -> Tuple[str, Dict]:
            """Fetch comprehensive data for a single symbol"""
            try:
                normalized_symbol = self.normalize_symbol(symbol)
                ticker = yf.Ticker(normalized_symbol)
                
                # Get current price and basic info
                hist = ticker.history(period="1d")
                info = ticker.info
                
                result = {
                    'current_price': None,
                    'info': {},
                    'historical_data': pd.DataFrame(),
                    'fundamental_metrics': {},
                    'error': None
                }
                
                # Current price
                if not hist.empty:
                    result['current_price'] = hist['Close'].iloc[-1]
                    
                    # Get more historical data for analysis
                    hist_extended = ticker.history(period="2y")
                    result['historical_data'] = hist_extended
                else:
                    logger.warning(f"No price data available for {symbol}")
                
                # Store info
                result['info'] = info
                
                # Calculate fundamental metrics if we have data
                if info and not result['historical_data'].empty:
                    result['fundamental_metrics'] = self.calculate_advanced_fundamental_metrics(
                        symbol, info, result['historical_data']
                    )
                
                logger.debug(f"Successfully fetched data for {symbol}")
                return symbol, result
                
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {e}")
                return symbol, {
                    'current_price': None,
                    'info': {},
                    'historical_data': pd.DataFrame(),
                    'fundamental_metrics': {},
                    'error': str(e)
                }
        
        # Fetch data concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(fetch_comprehensive_data, symbol): symbol for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                
                # Store the data
                self.current_prices[symbol] = data['current_price']
                self.stock_info[symbol] = data['info']
                self.historical_data[symbol] = data['historical_data']
                
                # Store fundamental metrics in holdings
                if symbol in self.holdings:
                    self.holdings[symbol]['fundamental_metrics'] = data['fundamental_metrics']
                    self.holdings[symbol]['info'] = data['info']
                    self.holdings[symbol]['historical_data'] = data['historical_data']
        
        # Fetch benchmark data (S&P 500)
        try:
            logger.info("Fetching benchmark data (S&P 500)...")
            spy = yf.Ticker("SPY")
            self.benchmark_data = spy.history(period="2y")
            logger.info("Benchmark data fetched successfully")
        except Exception as e:
            logger.warning(f"Error fetching benchmark data: {e}")
        
        # Summary
        successful_fetches = sum(1 for price in self.current_prices.values() if price is not None)
        logger.info(f"Market data fetch complete: {successful_fetches}/{len(symbols)} symbols successful")
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series"""
        try:
            peak = prices.expanding(min_periods=1).max()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        except:
            return 0.0

    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate comprehensive portfolio performance metrics"""
        logger.info("Calculating portfolio metrics...")
        
        metrics = {
            # Basic metrics
            'total_invested': 0,
            'current_value': 0,
            'total_return': 0,
            'total_return_pct': 0,
            'realized_gains': 0,
            'unrealized_gains': 0,
            'dividend_income': 0,
            
            # Holdings analysis
            'holdings_summary': [],
            'sector_allocation': {},
            'industry_allocation': {},
            'market_cap_allocation': {'Large': 0, 'Mid': 0, 'Small': 0, 'Unknown': 0},
            
            # Performance metrics
            'best_performer': {'symbol': '', 'return_pct': float('-inf')},
            'worst_performer': {'symbol': '', 'return_pct': float('inf')},
            'largest_holding': {'symbol': '', 'value': 0},
            
            # Risk metrics
            'portfolio_volatility': 0,
            'portfolio_beta': 0,
            'portfolio_var_95': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            
            # Diversification
            'diversification_score': 0,
            'concentration_risk': 0,
            'num_holdings': 0,
            'num_sectors': 0,
            
            # Time-based analysis
            'holding_periods': [],
            'monthly_returns': [],
            'annual_returns': {},
            
            # ESG and Quality scores (if available)
            'esg_score': 0,
            'quality_score': 0,
        }
        
        # Initialize dividend income and realized gains in holdings if not present
        for symbol in self.holdings:
            if 'dividend_income' not in self.holdings[symbol]:
                self.holdings[symbol]['dividend_income'] = 0
            if 'realized_pnl' not in self.holdings[symbol]:
                self.holdings[symbol]['realized_pnl'] = 0
            
            metrics['dividend_income'] += self.holdings[symbol]['dividend_income']
            metrics['realized_gains'] += self.holdings[symbol]['realized_pnl']
        
        # Process current holdings
        portfolio_weights = []
        portfolio_returns = []
        portfolio_betas = []
        
        for symbol, holding in self.holdings.items():
            if holding['quantity'] <= 0:
                continue
                
            current_price = self.current_prices.get(symbol, 0)
            current_value = holding['quantity'] * current_price
            cost_basis = holding['total_cost']
            unrealized_gain = current_value - cost_basis
            unrealized_gain_pct = (unrealized_gain / cost_basis * 100) if cost_basis > 0 else 0
            
            # Update totals
            metrics['total_invested'] += cost_basis
            metrics['current_value'] += current_value
            metrics['unrealized_gains'] += unrealized_gain
            
            # Track best/worst performers
            if unrealized_gain_pct > metrics['best_performer']['return_pct']:
                metrics['best_performer'] = {'symbol': symbol, 'return_pct': unrealized_gain_pct}
            if unrealized_gain_pct < metrics['worst_performer']['return_pct']:
                metrics['worst_performer'] = {'symbol': symbol, 'return_pct': unrealized_gain_pct}
            
            # Track largest holding
            if current_value > metrics['largest_holding']['value']:
                metrics['largest_holding'] = {'symbol': symbol, 'value': current_value}
            
            # Get enhanced stock data
            info = self.stock_info.get(symbol, {})
            hist_data = self.historical_data.get(symbol, pd.DataFrame())
            
            # Enhanced sector and industry allocation with better fallback
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # If sector info is missing, try symbol mapping
            if sector == 'Unknown' or not sector:
                sector = self._get_sector_from_symbol(symbol)
            
            # Same for industry
            if industry == 'Unknown' or not industry:
                industry = self._get_industry_from_symbol(symbol)
            
            if sector not in metrics['sector_allocation']:
                metrics['sector_allocation'][sector] = 0
            metrics['sector_allocation'][sector] += current_value
            
            if industry not in metrics['industry_allocation']:
                metrics['industry_allocation'][industry] = 0
            metrics['industry_allocation'][industry] += current_value
            
            # Market cap classification
            market_cap = info.get('marketCap', 0)
            if market_cap > 200e9:
                cap_class = 'Large'
            elif market_cap > 10e9:
                cap_class = 'Mid'
            elif market_cap > 0:
                cap_class = 'Small'
            else:
                cap_class = 'Unknown'
            metrics['market_cap_allocation'][cap_class] += current_value
            
            # Calculate holding period
            first_purchase = holding.get('first_purchase_date')
            holding_days = 0
            if first_purchase:
                holding_days = (datetime.now() - first_purchase).days
                metrics['holding_periods'].append(holding_days)
            
            # Create detailed holding summary
            weight = (current_value / metrics['current_value'] * 100) if metrics['current_value'] > 0 else 0
            
            # Get fundamental metrics if available
            fundamental_metrics = holding.get('fundamental_metrics', {})
            
            metrics['holdings_summary'].append({
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': sector,
                'industry': industry,
                'quantity': holding['quantity'],
                'avg_cost': holding['avg_cost'],
                'current_price': current_price,
                'current_value': current_value,
                'cost_basis': cost_basis,
                'total_cost': cost_basis,
                'unrealized_gain': unrealized_gain,
                'unrealized_gain_pct': unrealized_gain_pct,
                'realized_pnl': holding['realized_pnl'],
                'dividend_income': holding['dividend_income'],
                'weight': weight,
                'volatility': fundamental_metrics.get('volatility', 0),
                'beta': info.get('beta', 1.0),
                'pe_ratio': fundamental_metrics.get('pe_ratio'),
                'dividend_yield': fundamental_metrics.get('dividend_yield', 0),
                'market_cap': market_cap,
                'var_95': fundamental_metrics.get('var_95', 0),
                'max_drawdown': fundamental_metrics.get('max_drawdown', 0),
                'holding_period_days': holding_days
            })
            
            # Collect data for portfolio-level calculations
            if weight > 0:
                portfolio_weights.append(weight / 100)
                portfolio_betas.append(info.get('beta', 1.0))
                
                # Calculate returns if we have historical data
                if not hist_data.empty and len(hist_data) > 1:
                    returns = hist_data['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        portfolio_returns.append(returns)
        
        # Calculate overall performance
        metrics['total_return'] = metrics['unrealized_gains'] + metrics['realized_gains']
        metrics['total_return_pct'] = (metrics['total_return'] / metrics['total_invested'] * 100) if metrics['total_invested'] > 0 else 0
        
        # Portfolio-level risk metrics
        if portfolio_weights and portfolio_betas:
            metrics['portfolio_beta'] = np.average(portfolio_betas, weights=portfolio_weights)
            
            # Calculate portfolio volatility (simplified)
            if portfolio_returns:
                weighted_volatilities = []
                for i, returns in enumerate(portfolio_returns):
                    if len(returns) > 0:
                        vol = returns.std() * np.sqrt(252)
                        if i < len(portfolio_weights):
                            weighted_volatilities.append(vol * portfolio_weights[i])
                metrics['portfolio_volatility'] = sum(weighted_volatilities)
        
        # Risk-adjusted returns
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        if metrics['portfolio_volatility'] > 0:
            excess_return = metrics['total_return_pct'] / 100 - risk_free_rate
            metrics['sharpe_ratio'] = excess_return / metrics['portfolio_volatility']
        
        # Diversification metrics
        metrics['num_holdings'] = len([h for h in metrics['holdings_summary'] if h['quantity'] > 0])
        metrics['num_sectors'] = len(metrics['sector_allocation'])
        
        # Concentration risk (Herfindahl Index)
        weights = [h['weight'] / 100 for h in metrics['holdings_summary']]
        metrics['concentration_risk'] = sum(w**2 for w in weights) if weights else 0
        
        # Diversification score (0-100)
        holding_score = min(metrics['num_holdings'] / 20, 1.0) * 40  # Max 40 points for holdings
        sector_score = min(metrics['num_sectors'] / 10, 1.0) * 30   # Max 30 points for sectors
        concentration_score = max(0, (1 - metrics['concentration_risk']) * 30)  # Max 30 points for low concentration
        metrics['diversification_score'] = holding_score + sector_score + concentration_score
        
        # Sort holdings by value
        metrics['holdings_summary'].sort(key=lambda x: x['current_value'], reverse=True)
        
        return metrics
    
    def _get_sector_from_symbol(self, symbol: str) -> str:
        """Get sector from symbol using mapping"""
        sector_mapping = {
            'NVDA': 'Technology', 'PLTR': 'Technology', 'CRWD': 'Technology',
            'MSFT': 'Technology', 'GOOGL': 'Technology', 'INTU': 'Technology',
            'TSM': 'Technology', 'QCOM': 'Technology', 'AAPL': 'Technology',
            'AMZN': 'Consumer Cyclical', 'TSLA': 'Consumer Cyclical',
            'COST': 'Consumer Defensive', 'OXY': 'Energy',
            'BAC': 'Financial Services', 'BRK.B': 'Financial Services',
            'VOO': 'ETF', 'QQQ': 'ETF', 'VUG': 'ETF', 'SPY': 'ETF',
            'VTI': 'ETF', 'XLY': 'ETF', 'SMH': 'ETF', 'VYM': 'ETF',
            'SOXX': 'ETF', 'VHT': 'ETF', 'DLR': 'Real Estate'
        }
        return sector_mapping.get(symbol, 'Unknown')
    
    def _get_industry_from_symbol(self, symbol: str) -> str:
        """Get industry from symbol using mapping"""
        industry_mapping = {
            'NVDA': 'Semiconductors', 'PLTR': 'Software', 'CRWD': 'Software',
            'MSFT': 'Software', 'GOOGL': 'Internet Content & Information',
            'INTU': 'Software', 'TSM': 'Semiconductors', 'QCOM': 'Semiconductors',
            'AAPL': 'Consumer Electronics', 'AMZN': 'Internet Retail',
            'TSLA': 'Auto Manufacturers', 'COST': 'Discount Stores',
            'OXY': 'Oil & Gas E&P', 'BAC': 'Banks', 'BRK.B': 'Insurance',
            'VOO': 'ETF', 'QQQ': 'ETF', 'VUG': 'ETF', 'SPY': 'ETF',
            'VTI': 'ETF', 'XLY': 'ETF', 'SMH': 'ETF', 'VYM': 'ETF',
            'SOXX': 'ETF', 'VHT': 'ETF', 'DLR': 'REIT'
        }
        return industry_mapping.get(symbol, 'Unknown')
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for yfinance lookup"""
        return symbol.replace('.', '-')
    
    def safe_get_stock_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Safely get stock data with fallback"""
        try:
            ticker = yf.Ticker(self.normalize_symbol(symbol))
            return ticker.history(period=period)
        except:
            return pd.DataFrame()

def display_comprehensive_portfolio_analysis():
    """Display complete portfolio analysis in rich console format"""
    
    print("🚀" + "="*120)
    print("🎯 COMPREHENSIVE PORTFOLIO ANALYSIS SUITE")
    print("👤 Author: Divakar")
    print("📅 Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*120)
    
    try:
        print("📂 Loading portfolio data...")
        analyzer = EnhancedPortfolioAnalyzer("portfolio.csv")
        analyzer.load_data()
        
        print("🧮 Calculating holdings...")
        analyzer.calculate_holdings()
        
        print("📡 Fetching market data...")
        analyzer.get_market_data()
        
        print("📊 Computing portfolio metrics...")
        metrics = analyzer.calculate_portfolio_metrics()
        
        # Ensure we have the metrics
        if not metrics:
            raise Exception("Failed to calculate portfolio metrics")
        
        print("\n" + "✅" + "="*60)
        print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        print("="*60)
        
        # Display main metrics (like original console output)
        print(f"📊 Total Return: ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.1f}%)")
        print(f"💰 Current Value: ${metrics['current_value']:,.2f}")
        print(f"💵 Total Invested: ${metrics['total_invested']:,.2f}")
        print(f"📈 Unrealized Gains: ${metrics['unrealized_gains']:,.2f}")
        print(f"💸 Dividend Income: ${metrics['dividend_income']:,.2f}")
        print(f"🎯 Diversification Score: {metrics['diversification_score']:.0f}/100")
        print(f"📊 Portfolio Beta: {metrics['portfolio_beta']:.2f}")
        print(f"⚡ Portfolio Volatility: {metrics['portfolio_volatility']:.1f}%")
        print(f"🏆 Best Performer: {metrics['best_performer']['symbol']} ({metrics['best_performer']['return_pct']:.1f}%)")
        print(f"⚠️  Worst Performer: {metrics['worst_performer']['symbol']} ({metrics['worst_performer']['return_pct']:.1f}%)")
        
        print(f"\n📋 Analysis Details:")
        print(f"   🔍 Individual Stock Profiles: {len([h for h in metrics['holdings_summary'] if h['current_value'] > 1])} detailed analyses")
        print(f"   📊 Portfolio Contribution Analysis: Complete")
        print(f"   💡 Investment Recommendations: Available for all holdings")
        print(f"   📈 Total Holdings Analyzed: {metrics['num_holdings']} active positions")
        print(f"   🌐 Sectors Covered: {metrics['num_sectors']} different sectors")
        
        # Display detailed holdings table
        display_detailed_holdings_table(metrics)
        
        # Display strategic analysis
        display_strategic_analysis(metrics)
        
        # Display sector analysis
        display_sector_analysis(metrics)
        
        # Display risk analysis
        display_risk_analysis(metrics)
        
        # Display recommendations
        display_investment_recommendations(metrics)
        
        # Add comprehensive advanced analysis
        display_fundamental_analysis(metrics)
        display_performance_attribution_analysis(metrics)
        display_income_dividend_analysis(metrics)
        display_advanced_market_sector_analysis(metrics)
        display_portfolio_optimization_analysis(metrics)
        
        print("\n" + "🎉" + "="*120)
        print("🎉 COMPLETE ADVANCED PORTFOLIO ANALYSIS FINISHED!")
        print("="*120)
        print("📊 Comprehensive analysis including fundamentals, attribution, income, and optimization")
        print("🚀 All analysis displayed in rich console format")
        print("="*120)
        
        # Add disclaimer
        print("\n" + "⚠️" + "="*120)
        print("⚠️ IMPORTANT DISCLAIMER")
        print("="*120)
        print("📋 This analysis is for informational purposes only and should not be considered as")
        print("   investment advice. Past performance does not guarantee future results.")
        print("💡 Key Points:")
        print("   • All data is sourced from public APIs and may contain inaccuracies")
        print("   • Market conditions can change rapidly, affecting portfolio performance")
        print("   • This tool does not account for individual risk tolerance or financial goals")
        print("   • Consider consulting with a qualified financial advisor before making investment decisions")
        print("   • The author (Divakar) is not responsible for any investment losses")
        print("📞 Always do your own research and consider multiple sources before investing")
        print("="*120)
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        print("Using fallback console display...")
        display_fallback_portfolio_table()

def display_detailed_holdings_table(metrics):
    """Display detailed holdings table in console"""
    
    print("\n" + "📊" + "="*150)
    print("📊 PORTFOLIO GAINS & LOSSES ANALYSIS")
    print("="*150)
    print(f"📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*150)
    
    # Sort holdings by current value (descending)
    sorted_holdings = sorted(
        [h for h in metrics['holdings_summary'] if h['current_value'] > 0], 
        key=lambda x: x['current_value'], 
        reverse=True
    )
    
    total_value = sum(h['current_value'] for h in sorted_holdings)
    
    print("\n🏆 HOLDINGS RANKED BY CURRENT VALUE:")
    print("-"*150)
    
    # Header
    print(f"{'Rank':<4} {'Symbol':<8} {'Shares':<12} {'Avg Cost':<10} {'Current Price':<14} {'Current Value':<14} {'Gain/Loss':<14} {'Gain %':<10} {'Weight %':<10} {'Status'}")
    print("-"*150)
    
    # Data rows
    for i, holding in enumerate(sorted_holdings, 1):
        rank = i
        symbol = holding['symbol']
        shares = holding['quantity']
        avg_cost = holding['avg_cost']
        current_price = holding['current_price']
        current_value = holding['current_value']
        unrealized_gain = holding.get('unrealized_gain', 0)
        gain_pct = holding.get('unrealized_gain_pct', 0)
        weight = (current_value / total_value * 100) if total_value > 0 else 0
        
        # Status indicator
        if symbol in ['PLTR', 'NVDA', 'CRWD']:
            status = "🚀"
        elif symbol in ['VOO', 'QQQ', 'SPY']:
            status = "📈"
        else:
            status = "📊"
        
        print(f"{rank:<4} {symbol:<8} {shares:<12.4f} ${avg_cost:<9.2f} ${current_price:<13.2f} ${current_value:<13.2f} ${unrealized_gain:<13.2f} {gain_pct:<9.1f}% {weight:<9.1f}% {status}")
    
    # Totals row
    total_cost = sum(h.get('total_cost', 0) for h in sorted_holdings)
    total_gain = sum(h.get('unrealized_gain', 0) for h in sorted_holdings)
    total_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0
    
    print("-"*150)
    print(f"{'TOTAL':<4} {'':<8} {'':<12} {'':<10} {'':<14} ${total_value:<13.2f} ${total_gain:<13.2f} {total_gain_pct:<9.1f}% {'100.0%':<9}")
    print("="*150)

def display_strategic_analysis(metrics):
    """Display strategic portfolio analysis"""
    
    print("\n" + "🎯" + "="*120)
    print("🎯 STRATEGIC PORTFOLIO ANALYSIS & RECOMMENDATIONS")
    print("="*120)
    
    sorted_holdings = sorted(
        [h for h in metrics['holdings_summary'] if h['current_value'] > 0], 
        key=lambda x: x['current_value'], 
        reverse=True
    )
    
    total_value = sum(h['current_value'] for h in sorted_holdings)
    
    # Calculate strategic metrics
    tech_symbols = ['NVDA', 'PLTR', 'CRWD', 'MSFT', 'GOOGL', 'INTU', 'TSM', 'QCOM']
    etf_symbols = ['VOO', 'QQQ', 'VUG', 'SPY', 'XLY', 'VTI', 'SMH', 'VYM', 'SOXX', 'VHT', 'DLR']
    growth_symbols = ['PLTR', 'NVDA', 'CRWD', 'TSLA']
    
    tech_value = sum(h['current_value'] for h in sorted_holdings if h['symbol'] in tech_symbols)
    etf_value = sum(h['current_value'] for h in sorted_holdings if h['symbol'] in etf_symbols)
    growth_value = sum(h['current_value'] for h in sorted_holdings if h['symbol'] in growth_symbols)
    
    tech_pct = (tech_value / total_value * 100)
    etf_pct = (etf_value / total_value * 100)
    growth_pct = (growth_value / total_value * 100)
    
    print(f"\n📊 STRATEGIC POSITIONING ANALYSIS:")
    print(f"   🔹 Technology Exposure: {tech_pct:.1f}% (${tech_value:,.0f})")
    print(f"     • Assessment: {'High' if tech_pct > 40 else 'Moderate' if tech_pct > 25 else 'Conservative'} technology weighting")
    print(f"     • Risk/Reward: Positioned for digital transformation but with concentration risk")
    
    print(f"\n   🔹 Core Market Exposure: {etf_pct:.1f}% (${etf_value:,.0f})")
    print(f"     • Assessment: {'Well-diversified' if etf_pct > 40 else 'Moderate' if etf_pct > 25 else 'Growth-focused'} core allocation")
    print(f"     • Stability Factor: Broad market ETFs provide portfolio stability")
    
    print(f"\n   🔹 Growth Stock Concentration: {growth_pct:.1f}% (${growth_value:,.0f})")
    print(f"     • Assessment: {'Aggressive' if growth_pct > 30 else 'Moderate' if growth_pct > 15 else 'Conservative'} growth allocation")
    print(f"     • Volatility Impact: High-growth names drive performance and volatility")
    
    print(f"\n🔄 REBALANCING RECOMMENDATIONS:")
    if tech_pct > 50:
        print("   • ⚠️  Consider reducing technology concentration for risk management")
    elif tech_pct < 20:
        print("   • 📈 Consider increasing technology exposure for growth potential")
    else:
        print("   • ✅ Technology allocation appears balanced")
    
    if etf_pct < 30:
        print("   • 📊 Consider increasing broad market ETF allocation for stability")
    else:
        print("   • ✅ Good diversification through ETF holdings")
    
    if growth_pct > 35:
        print("   • ⚠️  High growth concentration - monitor for volatility management")
    else:
        print("   • ✅ Growth allocation within reasonable range")
    
    print(f"\n📈 MARKET ENVIRONMENT CONSIDERATIONS:")
    print(f"   • Interest Rate Sensitivity: Technology positions may be sensitive to rate changes")
    print(f"   • Economic Cycle Positioning: Portfolio favors growth over value")
    print(f"   • Inflation Protection: Limited commodities/energy exposure")
    print(f"   • Dividend Income: ${sum(h.get('dividend_income', 0) for h in sorted_holdings):,.2f} annual potential")

def display_sector_analysis(metrics):
    """Display detailed sector analysis"""
    
    print("\n" + "📊" + "="*120)
    print("📊 SECTOR ALLOCATION & ANALYSIS")
    print("="*120)
    
    sorted_holdings = sorted(
        [h for h in metrics['holdings_summary'] if h['current_value'] > 0], 
        key=lambda x: x['current_value'], 
        reverse=True
    )
    
    total_value = sum(h['current_value'] for h in sorted_holdings)
    
    # Enhanced sector mapping
    sector_mapping = {
        'NVDA': 'Technology - Semiconductors', 'PLTR': 'Technology - Software', 'CRWD': 'Technology - Cybersecurity',
        'MSFT': 'Technology - Software', 'GOOGL': 'Technology - Internet', 'INTU': 'Technology - Software',
        'TSM': 'Technology - Semiconductors', 'QCOM': 'Technology - Semiconductors',
        'AMZN': 'Consumer Discretionary - E-commerce', 'TSLA': 'Consumer Discretionary - Electric Vehicles',
        'COST': 'Consumer Staples - Retail', 'OXY': 'Energy - Oil & Gas',
        'VOO': 'Broad Market ETF - S&P 500', 'QQQ': 'Technology ETF - NASDAQ 100', 
        'VUG': 'Growth ETF', 'SPY': 'Broad Market ETF - S&P 500',
        'VTI': 'Total Market ETF', 'XLY': 'Consumer Discretionary ETF',
        'SMH': 'Semiconductor ETF', 'VYM': 'Dividend ETF', 'SOXX': 'Semiconductor ETF',
        'VHT': 'Healthcare ETF', 'DLR': 'Real Estate Investment Trust', 'AAPL': 'Technology - Hardware',
        'BAC': 'Financials - Banking'
    }
    
    # Calculate sector breakdown
    sector_breakdown = defaultdict(lambda: {'value': 0, 'holdings': []})
    
    for holding in sorted_holdings:
        sector = sector_mapping.get(holding['symbol'], 'Other')
        sector_breakdown[sector]['value'] += holding['current_value']
        sector_breakdown[sector]['holdings'].append(holding['symbol'])
    
    # Sort sectors by value
    sorted_sectors = sorted(sector_breakdown.items(), key=lambda x: x[1]['value'], reverse=True)
    
    print("\n🏢 SECTOR BREAKDOWN:")
    print("-"*120)
    print(f"{'Sector/Category':<40} {'Allocation':<12} {'Value':<15} {'Holdings':<30} {'Assessment'}")
    print("-"*120)
    
    for sector, data in sorted_sectors:
        allocation_pct = (data['value'] / total_value * 100)
        holdings_str = ', '.join(data['holdings'][:4])
        if len(data['holdings']) > 4:
            holdings_str += f" (+{len(data['holdings'])-4})"
        
        # Strategic assessment
        if 'Technology' in sector:
            assessment = "High Growth Potential"
        elif 'ETF' in sector or 'Broad Market' in sector:
            assessment = "Diversification & Stability"
        elif 'Dividend' in sector:
            assessment = "Income Generation"
        elif 'Energy' in sector:
            assessment = "Inflation Hedge"
        else:
            assessment = "Sector Exposure"
        
        print(f"{sector:<40} {allocation_pct:>6.1f}%     ${data['value']:>10,.0f}  {holdings_str:<30} {assessment}")
    
    print("="*120)

def display_risk_analysis(metrics):
    """Display risk analysis"""
    
    print("\n" + "⚠️" + "="*120)
    print("⚠️ RISK ANALYSIS & ASSESSMENT")
    print("="*120)
    
    print(f"\n📊 RISK METRICS:")
    print(f"   • Portfolio Volatility: {metrics['portfolio_volatility']:.1f}% ({'High' if metrics['portfolio_volatility'] > 25 else 'Medium' if metrics['portfolio_volatility'] > 15 else 'Low'} risk)")
    print(f"   • Beta vs Market: {metrics['portfolio_beta']:.2f} ({'More volatile' if metrics['portfolio_beta'] > 1.1 else 'Less volatile' if metrics['portfolio_beta'] < 0.9 else 'Similar to market'})")
    print(f"   • Concentration Risk: {metrics.get('concentration_risk', 0):.3f} ({'High' if metrics.get('concentration_risk', 0) > 0.2 else 'Medium' if metrics.get('concentration_risk', 0) > 0.1 else 'Low'})")
    print(f"   • Sector Diversity: {metrics['num_sectors']} sectors ({'Good' if metrics['num_sectors'] >= 5 else 'Limited'})")
    print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.2f} ({'Excellent' if metrics['sharpe_ratio'] > 1 else 'Good' if metrics['sharpe_ratio'] > 0.5 else 'Below average'})")
    
    sorted_holdings = sorted(
        [h for h in metrics['holdings_summary'] if h['current_value'] > 0], 
        key=lambda x: x['current_value'], 
        reverse=True
    )
    
    # Top 3 concentration
    top_3_value = sum(h['current_value'] for h in sorted_holdings[:3])
    total_value = sum(h['current_value'] for h in sorted_holdings)
    concentration = (top_3_value / total_value * 100)
    
    print(f"\n🎯 CONCENTRATION ANALYSIS:")
    print(f"   • Top 3 Holdings: {concentration:.1f}% of portfolio")
    print(f"   • Risk Level: {'High' if concentration > 60 else 'Medium' if concentration > 40 else 'Low'} concentration")
    print(f"   • Recommendation: {'Consider diversification' if concentration > 60 else 'Well diversified' if concentration < 40 else 'Moderate concentration'}")

def display_investment_recommendations(metrics):
    """Display investment recommendations"""
    
    print("\n" + "💡" + "="*120)
    print("💡 INVESTMENT RECOMMENDATIONS & ACTION ITEMS")
    print("="*120)
    
    sorted_holdings = sorted(
        [h for h in metrics['holdings_summary'] if h['current_value'] > 0], 
        key=lambda x: x['current_value'], 
        reverse=True
    )
    
    print(f"\n🎯 STRATEGIC ACTION ITEMS:")
    print(f"   1. 📊 Monitor technology sector rotation and valuation metrics")
    print(f"   2. 🌍 Consider adding international diversification if not present elsewhere") 
    print(f"   3. 🛡️  Evaluate defensive positions for market downturns")
    print(f"   4. ⚖️  Review position sizing relative to conviction levels")
    print(f"   5. 💰 Assess tax implications of any rebalancing activities")
    print(f"   6. 📈 Consider dollar-cost averaging for volatile positions")
    print(f"   7. 🔄 Review and rebalance quarterly based on performance")
    
    print(f"\n🏆 TOP PERFORMERS TO WATCH:")
    winners = [h for h in sorted_holdings if h.get('unrealized_gain_pct', 0) > 10]
    for holding in winners[:5]:
        print(f"   • {holding['symbol']}: {holding.get('unrealized_gain_pct', 0):.1f}% gain - Consider taking profits or letting it ride")
    
    print(f"\n⚠️  POSITIONS NEEDING ATTENTION:")
    losers = [h for h in sorted_holdings if h.get('unrealized_gain_pct', 0) < -5]
    for holding in losers[:3]:
        print(f"   • {holding['symbol']}: {holding.get('unrealized_gain_pct', 0):.1f}% loss - Evaluate for potential tax loss harvesting or hold")
    
    print(f"\n📊 PORTFOLIO OPTIMIZATION SUGGESTIONS:")
    total_value = sum(h['current_value'] for h in sorted_holdings)
    etf_value = sum(h['current_value'] for h in sorted_holdings if h['symbol'] in ['VOO', 'QQQ', 'VUG', 'SPY', 'XLY', 'VTI', 'SMH', 'VYM', 'SOXX', 'VHT', 'DLR'])
    etf_pct = (etf_value / total_value * 100)
    
    if etf_pct < 30:
        print(f"   • Consider increasing ETF allocation for better diversification")
    elif etf_pct > 70:
        print(f"   • Consider adding selective individual stock positions for alpha generation")
    else:
        print(f"   • Good balance between ETFs ({etf_pct:.1f}%) and individual stocks")

def display_fallback_portfolio_table():
    """Fallback console display if main analyzer fails"""
    
    print("\n🔄 Running fallback portfolio analysis...")
    
    # Get basic portfolio data
    portfolio_data = [
        {'symbol': 'VOO', 'shares': 2.94, 'description': 'Vanguard S&P 500 ETF'},
        {'symbol': 'QQQ', 'shares': 1.43, 'description': 'Invesco QQQ Trust'},
        {'symbol': 'CRWD', 'shares': 1.13, 'description': 'CrowdStrike Holdings'},
        {'symbol': 'PLTR', 'shares': 3.04, 'description': 'Palantir Technologies'},
        {'symbol': 'NVDA', 'shares': 1.97, 'description': 'NVIDIA Corporation'},
        {'symbol': 'VUG', 'shares': 0.65, 'description': 'Vanguard Growth ETF'},
        {'symbol': 'MSFT', 'shares': 0.32, 'description': 'Microsoft Corporation'},
        {'symbol': 'DLR', 'shares': 0.70, 'description': 'Digital Realty Trust'},
        {'symbol': 'TSM', 'shares': 0.48, 'description': 'Taiwan Semiconductor'},
        {'symbol': 'SPY', 'shares': 0.15, 'description': 'SPDR S&P 500 ETF'},
        {'symbol': 'AMZN', 'shares': 0.27, 'description': 'Amazon.com Inc'},
        {'symbol': 'TSLA', 'shares': 0.19, 'description': 'Tesla Inc'},
        {'symbol': 'XLY', 'shares': 0.23, 'description': 'Consumer Discretionary SPDR'},
        {'symbol': 'VTI', 'shares': 0.11, 'description': 'Vanguard Total Stock Market'},
        {'symbol': 'COST', 'shares': 0.03, 'description': 'Costco Wholesale'},
        {'symbol': 'SMH', 'shares': 0.09, 'description': 'VanEck Semiconductor ETF'},
        {'symbol': 'OXY', 'shares': 0.58, 'description': 'Occidental Petroleum'},
        {'symbol': 'INTU', 'shares': 0.03, 'description': 'Intuit Inc'},
        {'symbol': 'GOOGL', 'shares': 0.13, 'description': 'Alphabet Inc'},
        {'symbol': 'VYM', 'shares': 0.16, 'description': 'Vanguard High Dividend Yield'},
        {'symbol': 'SOXX', 'shares': 0.09, 'description': 'iShares Semiconductor ETF'},
        {'symbol': 'QCOM', 'shares': 0.12, 'description': 'Qualcomm Inc'},
        {'symbol': 'VHT', 'shares': 0.07, 'description': 'Vanguard Health Care ETF'}
    ]
    
    print("🔄 Fetching current market prices...")
    
    # Get current prices and display
    results = []
    total_value = 0
    
    for stock in portfolio_data:
        symbol = stock['symbol']
        shares = stock['shares']
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                current_value = shares * current_price
                total_value += current_value
                
                results.append({
                    'symbol': symbol,
                    'shares': shares,
                    'current_price': current_price,
                    'current_value': current_value,
                    'description': stock['description']
                })
            else:
                print(f"⚠️  No data available for {symbol}")
                
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
    
    # Sort and display
    results.sort(key=lambda x: x['current_value'], reverse=True)
    
    print("\n" + "📊" + "="*150)
    print("📊 BASIC PORTFOLIO ANALYSIS")
    print("="*150)
    
    print(f"{'Rank':<4} {'Symbol':<8} {'Shares':<12} {'Current Price':<14} {'Current Value':<14} {'Weight %':<10} {'Description':<40}")
    print("-"*150)
    
    for i, holding in enumerate(results, 1):
        weight = (holding['current_value'] / total_value * 100) if total_value > 0 else 0
        print(f"{i:<4} {holding['symbol']:<8} {holding['shares']:<12.4f} ${holding['current_price']:<13.2f} ${holding['current_value']:<13.2f} {weight:<9.1f}% {holding['description'][:38]}")
    
    print("-"*150)
    print(f"{'TOTAL':<4} {'':<8} {'':<12} {'':<14} ${total_value:<13.2f} {'100.0%':<10}")
    print("="*150)

def get_fundamental_data(symbol):
    """Get fundamental data for a stock"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Key fundamental metrics
        fundamentals = {
            'pe_ratio': info.get('trailingPE', None),
            'forward_pe': info.get('forwardPE', None),
            'peg_ratio': info.get('pegRatio', None),
            'price_to_book': info.get('priceToBook', None),
            'price_to_sales': info.get('priceToSalesTrailing12Months', None),
            'debt_to_equity': info.get('debtToEquity', None),
            'return_on_equity': info.get('returnOnEquity', None),
            'return_on_assets': info.get('returnOnAssets', None),
            'profit_margin': info.get('profitMargins', None),
            'operating_margin': info.get('operatingMargins', None),
            'revenue_growth': info.get('revenueGrowth', None),
            'earnings_growth': info.get('earningsGrowth', None),
            'current_ratio': info.get('currentRatio', None),
            'quick_ratio': info.get('quickRatio', None),
            'dividend_yield': info.get('dividendYield', None),
            'payout_ratio': info.get('payoutRatio', None),
            'beta': info.get('beta', None),
            'market_cap': info.get('marketCap', None),
            'enterprise_value': info.get('enterpriseValue', None),
            'book_value': info.get('bookValue', None),
            'cash_per_share': info.get('totalCashPerShare', None),
            'revenue_per_share': info.get('revenuePerShare', None),
            'free_cash_flow': info.get('freeCashflow', None),
            'operating_cash_flow': info.get('operatingCashflow', None),
            'recommendation': info.get('recommendationKey', None),
            'target_price': info.get('targetMeanPrice', None),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),
            'analyst_price_targets': {
                'low': info.get('targetLowPrice', None),
                'mean': info.get('targetMeanPrice', None),
                'high': info.get('targetHighPrice', None)
            }
        }
        
        return fundamentals
    except Exception as e:
        print(f"   ⚠️  Error getting fundamentals for {symbol}: {e}")
        return None

def display_fundamental_analysis(metrics):
    """Display comprehensive fundamental analysis"""
    
    print("\n" + "🔍" + "="*120)
    print("🔍 FUNDAMENTAL ANALYSIS & STOCK VALUATION")
    print("="*120)
    
    sorted_holdings = sorted(
        [h for h in metrics['holdings_summary'] if h['current_value'] > 0], 
        key=lambda x: x['current_value'], 
        reverse=True
    )
    
    # Filter for individual stocks (exclude ETFs)
    etf_symbols = ['VOO', 'QQQ', 'VUG', 'SPY', 'XLY', 'VTI', 'SMH', 'VYM', 'SOXX', 'VHT', 'DLR']
    individual_stocks = [h for h in sorted_holdings if h['symbol'] not in etf_symbols]
    
    print(f"\n📊 ANALYZING {len(individual_stocks)} INDIVIDUAL STOCKS FOR FUNDAMENTAL METRICS:")
    print("-"*120)
    
    fundamental_summary = {
        'undervalued': [],
        'fairly_valued': [],
        'overvalued': [],
        'high_growth': [],
        'dividend_stocks': [],
        'quality_stocks': []
    }
    
    for holding in individual_stocks[:10]:  # Top 10 by value
        symbol = holding['symbol']
        current_value = holding['current_value']
        weight = (current_value / sum(h['current_value'] for h in sorted_holdings) * 100)
        
        print(f"\n🔍 {symbol} - Fundamental Analysis (Weight: {weight:.1f}%):")
        print("   " + "-"*60)
        
        fundamentals = get_fundamental_data(symbol)
        
        if fundamentals:
            # Valuation Metrics
            print(f"   💰 VALUATION METRICS:")
            pe = fundamentals['pe_ratio']
            pb = fundamentals['price_to_book']
            ps = fundamentals['price_to_sales']
            peg = fundamentals['peg_ratio']
            
            if pe:
                pe_assessment = "Undervalued" if pe < 15 else "Fairly Valued" if pe < 25 else "Overvalued"
                print(f"      • P/E Ratio: {pe:.2f} ({pe_assessment})")
            if pb:
                pb_assessment = "Undervalued" if pb < 1.5 else "Fairly Valued" if pb < 3 else "Overvalued"
                print(f"      • P/B Ratio: {pb:.2f} ({pb_assessment})")
            if ps:
                print(f"      • P/S Ratio: {ps:.2f}")
            if peg:
                peg_assessment = "Undervalued" if peg < 1 else "Fairly Valued" if peg < 1.5 else "Overvalued"
                print(f"      • PEG Ratio: {peg:.2f} ({peg_assessment})")
            
            # Profitability & Growth
            print(f"   📈 PROFITABILITY & GROWTH:")
            roe = fundamentals['return_on_equity']
            profit_margin = fundamentals['profit_margin']
            revenue_growth = fundamentals['revenue_growth']
            earnings_growth = fundamentals['earnings_growth']
            
            if roe:
                roe_assessment = "Excellent" if roe > 0.15 else "Good" if roe > 0.10 else "Poor"
                print(f"      • ROE: {roe*100:.1f}% ({roe_assessment})")
            if profit_margin:
                print(f"      • Profit Margin: {profit_margin*100:.1f}%")
            if revenue_growth:
                print(f"      • Revenue Growth: {revenue_growth*100:.1f}%")
            if earnings_growth:
                growth_assessment = "High Growth" if earnings_growth > 0.20 else "Moderate Growth" if earnings_growth > 0.10 else "Low Growth"
                print(f"      • Earnings Growth: {earnings_growth*100:.1f}% ({growth_assessment})")
                if earnings_growth > 0.15:
                    fundamental_summary['high_growth'].append(symbol)
            
            # Financial Health
            print(f"   💪 FINANCIAL HEALTH:")
            debt_equity = fundamentals['debt_to_equity']
            current_ratio = fundamentals['current_ratio']
            quick_ratio = fundamentals['quick_ratio']
            
            if debt_equity:
                debt_assessment = "Low Debt" if debt_equity < 0.3 else "Moderate Debt" if debt_equity < 0.6 else "High Debt"
                print(f"      • Debt/Equity: {debt_equity:.2f} ({debt_assessment})")
            if current_ratio:
                liquidity_assessment = "Strong" if current_ratio > 2 else "Adequate" if current_ratio > 1.2 else "Weak"
                print(f"      • Current Ratio: {current_ratio:.2f} ({liquidity_assessment})")
            
            # Dividend Analysis
            div_yield = fundamentals['dividend_yield']
            payout_ratio = fundamentals['payout_ratio']
            if div_yield and div_yield > 0.02:
                print(f"   💰 DIVIDEND METRICS:")
                print(f"      • Dividend Yield: {div_yield*100:.2f}%")
                if payout_ratio:
                    payout_assessment = "Sustainable" if payout_ratio < 0.6 else "Moderate Risk" if payout_ratio < 0.8 else "High Risk"
                    print(f"      • Payout Ratio: {payout_ratio*100:.1f}% ({payout_assessment})")
                fundamental_summary['dividend_stocks'].append(symbol)
            
            # Analyst Recommendations
            recommendation = fundamentals['recommendation']
            target_price = fundamentals['target_price']
            current_price = holding['current_price']
            
            if recommendation or target_price:
                print(f"   🎯 ANALYST OUTLOOK:")
                if recommendation:
                    print(f"      • Recommendation: {recommendation.upper()}")
                if target_price and current_price:
                    upside = ((target_price - current_price) / current_price * 100)
                    upside_assessment = "Strong Upside" if upside > 20 else "Moderate Upside" if upside > 10 else "Limited Upside" if upside > 0 else "Potential Downside"
                    print(f"      • Target Price: ${target_price:.2f} (Current: ${current_price:.2f})")
                    print(f"      • Potential Upside: {upside:.1f}% ({upside_assessment})")
            
            # Overall Assessment
            print(f"   ⭐ OVERALL ASSESSMENT:")
            quality_score = 0
            valuation_score = 0
            
            # Quality scoring
            if roe and roe > 0.15: quality_score += 1
            if profit_margin and profit_margin > 0.10: quality_score += 1
            if debt_equity and debt_equity < 0.5: quality_score += 1
            if current_ratio and current_ratio > 1.5: quality_score += 1
            
            # Valuation scoring
            if pe and pe < 20: valuation_score += 1
            if pb and pb < 2: valuation_score += 1
            if peg and peg < 1.2: valuation_score += 1
            
            quality_rating = "High Quality" if quality_score >= 3 else "Medium Quality" if quality_score >= 2 else "Lower Quality"
            valuation_rating = "Undervalued" if valuation_score >= 2 else "Fairly Valued" if valuation_score >= 1 else "Overvalued"
            
            print(f"      • Quality Rating: {quality_rating} ({quality_score}/4)")
            print(f"      • Valuation Rating: {valuation_rating} ({valuation_score}/3)")
            
            # Categorize for summary
            if quality_score >= 3:
                fundamental_summary['quality_stocks'].append(symbol)
            if valuation_score >= 2:
                fundamental_summary['undervalued'].append(symbol)
            elif valuation_score == 1:
                fundamental_summary['fairly_valued'].append(symbol)
            else:
                fundamental_summary['overvalued'].append(symbol)
        
        time.sleep(0.5)  # Rate limiting
    
    # Summary recommendations
    print(f"\n📋 FUNDAMENTAL ANALYSIS SUMMARY:")
    print("-"*120)
    
    if fundamental_summary['undervalued']:
        print(f"   💎 UNDERVALUED OPPORTUNITIES: {', '.join(fundamental_summary['undervalued'])}")
        print(f"      → Consider increasing positions or holding for value appreciation")
    
    if fundamental_summary['overvalued']:
        print(f"   ⚠️  OVERVALUED POSITIONS: {', '.join(fundamental_summary['overvalued'])}")
        print(f"      → Consider taking profits or reducing position sizes")
    
    if fundamental_summary['high_growth']:
        print(f"   🚀 HIGH GROWTH STOCKS: {', '.join(fundamental_summary['high_growth'])}")
        print(f"      → Monitor for sustained growth but watch valuations")
    
    if fundamental_summary['quality_stocks']:
        print(f"   ⭐ HIGH QUALITY STOCKS: {', '.join(fundamental_summary['quality_stocks'])}")
        print(f"      → Core holdings with strong fundamentals - consider long-term holds")
    
    if fundamental_summary['dividend_stocks']:
        print(f"   💰 DIVIDEND INCOME STOCKS: {', '.join(fundamental_summary['dividend_stocks'])}")
        print(f"      → Income-generating assets - monitor payout sustainability")

def display_performance_attribution_analysis(metrics):
    """Display performance attribution analysis"""
    
    print("\n" + "📊" + "="*120)
    print("📊 PERFORMANCE & ATTRIBUTION ANALYSIS")
    print("="*120)
    
    sorted_holdings = sorted(
        [h for h in metrics['holdings_summary'] if h['current_value'] > 0], 
        key=lambda x: x['current_value'], 
        reverse=True
    )
    
    total_value = sum(h['current_value'] for h in sorted_holdings)
    total_cost = sum(h.get('total_cost', 0) for h in sorted_holdings)
    total_gain = sum(h.get('unrealized_gain', 0) for h in sorted_holdings)
    
    print(f"\n🎯 PORTFOLIO PERFORMANCE ATTRIBUTION:")
    print("-"*120)
    
    # Calculate contribution to total return
    print(f"{'Stock':<8} {'Weight %':<10} {'Return %':<12} {'Contribution':<15} {'Attribution':<15} {'Performance':<20}")
    print("-"*120)
    
    performance_leaders = []
    performance_laggards = []
    
    for holding in sorted_holdings:
        symbol = holding['symbol']
        weight = (holding['current_value'] / total_value * 100)
        return_pct = holding.get('unrealized_gain_pct', 0)
        gain = holding.get('unrealized_gain', 0)
        
        # Contribution to portfolio return
        contribution = (weight / 100) * (return_pct / 100) * 100
        
        # Attribution analysis
        if gain > 0:
            attribution = f"+${gain:,.0f}"
            performance_status = "Contributing ✅"
            if return_pct > 20:
                performance_leaders.append((symbol, return_pct, contribution))
        else:
            attribution = f"-${abs(gain):,.0f}"
            performance_status = "Detracting ❌"
            if return_pct < -10:
                performance_laggards.append((symbol, return_pct, contribution))
        
        print(f"{symbol:<8} {weight:<10.1f} {return_pct:<12.1f} {contribution:<15.2f} {attribution:<15} {performance_status:<20}")
    
    print("-"*120)
    total_contribution = sum((h['current_value'] / total_value) * (h.get('unrealized_gain_pct', 0) / 100) for h in sorted_holdings) * 100
    print(f"{'TOTAL':<8} {'100.0':<10} {(total_gain/total_cost)*100 if total_cost > 0 else 0:<12.1f} {total_contribution:<15.2f} ${total_gain:,.0f}")
    
    # Performance insights
    print(f"\n🏆 PERFORMANCE INSIGHTS:")
    print("-"*120)
    
    if performance_leaders:
        print(f"   📈 TOP CONTRIBUTORS TO RETURNS:")
        for symbol, return_pct, contribution in sorted(performance_leaders, key=lambda x: x[2], reverse=True)[:5]:
            print(f"      • {symbol}: {return_pct:.1f}% return contributing {contribution:.2f}% to portfolio")
    
    if performance_laggards:
        print(f"\n   📉 PERFORMANCE DETRACTORS:")
        for symbol, return_pct, contribution in sorted(performance_laggards, key=lambda x: x[2])[:3]:
            print(f"      • {symbol}: {return_pct:.1f}% return detracting {abs(contribution):.2f}% from portfolio")
    
    # Risk-adjusted performance
    print(f"\n⚖️  RISK-ADJUSTED PERFORMANCE ANALYSIS:")
    volatility = metrics.get('portfolio_volatility', 0)
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    beta = metrics.get('portfolio_beta', 1)
    
    print(f"   • Portfolio Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"   • Risk-Adjusted Return: {(total_gain/total_cost)/volatility*100 if volatility > 0 and total_cost > 0 else 0:.2f}")
    print(f"   • Beta-Adjusted Return: {((total_gain/total_cost)*100/beta) if beta > 0 and total_cost > 0 else 0:.2f}%")
    
    # Time-based attribution (if possible)
    print(f"\n📅 PERFORMANCE TIMING ANALYSIS:")
    recent_performers = []
    for holding in sorted_holdings:
        symbol = holding['symbol']
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            if len(hist) > 0:
                month_return = ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0] * 100)
                recent_performers.append((symbol, month_return))
        except:
            continue
    
    if recent_performers:
        recent_performers.sort(key=lambda x: x[1], reverse=True)
        print(f"   📈 BEST 1-MONTH PERFORMERS:")
        for symbol, month_return in recent_performers[:5]:
            print(f"      • {symbol}: {month_return:.1f}% (last 30 days)")
        
        print(f"\n   📉 WORST 1-MONTH PERFORMERS:")
        for symbol, month_return in recent_performers[-3:]:
            print(f"      • {symbol}: {month_return:.1f}% (last 30 days)")

def display_income_dividend_analysis(metrics):
    """Display comprehensive income and dividend analysis"""
    
    print("\n" + "💰" + "="*120)
    print("💰 INCOME & DIVIDEND ANALYSIS")
    print("="*120)
    
    sorted_holdings = sorted(
        [h for h in metrics['holdings_summary'] if h['current_value'] > 0], 
        key=lambda x: x['current_value'], 
        reverse=True
    )
    
    total_value = sum(h['current_value'] for h in sorted_holdings)
    total_dividend_income = sum(h.get('dividend_income', 0) for h in sorted_holdings)
    
    print(f"\n💸 DIVIDEND INCOME BREAKDOWN:")
    print("-"*120)
    print(f"{'Symbol':<8} {'Weight %':<10} {'Div Yield %':<12} {'Annual Div*':<15} {'Div Income':<15} {'Payout Ratio':<15} {'Status'}")
    print("-"*120)
    
    dividend_stocks = []
    high_yield_stocks = []
    dividend_growers = []
    
    for holding in sorted_holdings:
        symbol = holding['symbol']
        weight = (holding['current_value'] / total_value * 100)
        dividend_income = holding.get('dividend_income', 0)
        current_price = holding['current_price']
        
        # Get dividend data
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            div_yield = info.get('dividendYield', 0) or 0
            forward_div = info.get('dividendRate', 0) or 0
            payout_ratio = info.get('payoutRatio', 0) or 0
            
            if div_yield > 0.01:  # Only show stocks with meaningful dividends
                estimated_annual = holding['quantity'] * forward_div if forward_div else 0
                
                yield_category = "High Yield" if div_yield > 0.04 else "Medium Yield" if div_yield > 0.02 else "Low Yield"
                if div_yield > 0.04:
                    high_yield_stocks.append((symbol, div_yield))
                
                payout_status = "Sustainable" if payout_ratio < 0.6 else "Moderate" if payout_ratio < 0.8 else "High Risk"
                
                dividend_stocks.append({
                    'symbol': symbol,
                    'yield': div_yield,
                    'annual_estimate': estimated_annual,
                    'weight': weight
                })
                
                print(f"{symbol:<8} {weight:<10.1f} {div_yield*100:<12.2f} ${estimated_annual:<14.2f} ${dividend_income:<14.2f} {payout_ratio*100:<14.1f}% {yield_category}")
        
        except Exception as e:
            if dividend_income > 0:
                print(f"{symbol:<8} {weight:<10.1f} {'N/A':<12} {'N/A':<15} ${dividend_income:<14.2f} {'N/A':<15} {'Income Stock'}")
    
    print("-"*120)
    total_estimated_annual = sum(d['annual_estimate'] for d in dividend_stocks)
    portfolio_yield = (total_estimated_annual / total_value * 100) if total_value > 0 else 0
    
    print(f"{'TOTAL':<8} {'100.0':<10} {portfolio_yield:<12.2f} ${total_estimated_annual:<14.2f} ${total_dividend_income:<14.2f}")
    
    # Income analysis summary
    print(f"\n📈 INCOME ANALYSIS SUMMARY:")
    print("-"*120)
    
    print(f"   💰 Current Portfolio Yield: {portfolio_yield:.2f}%")
    print(f"   💸 Estimated Annual Dividend Income: ${total_estimated_annual:,.2f}")
    print(f"   📊 Actual Dividend Income (YTD): ${total_dividend_income:,.2f}")
    print(f"   🎯 Monthly Income Estimate: ${total_estimated_annual/12:,.2f}")
    
    # Dividend stock categorization
    if high_yield_stocks:
        print(f"\n🎯 HIGH DIVIDEND YIELD STOCKS (>4%):")
        for symbol, yield_pct in sorted(high_yield_stocks, key=lambda x: x[1], reverse=True):
            print(f"      • {symbol}: {yield_pct*100:.2f}% yield")
    
    # Income diversification
    dividend_weight = sum(d['weight'] for d in dividend_stocks)
    print(f"\n📊 INCOME DIVERSIFICATION:")
    print(f"   • Dividend-Paying Holdings: {len(dividend_stocks)} stocks")
    print(f"   • Dividend Weight in Portfolio: {dividend_weight:.1f}%")
    print(f"   • Non-Dividend Weight: {100-dividend_weight:.1f}%")
    
    if dividend_weight < 30:
        print(f"   💡 Suggestion: Consider increasing dividend allocation for more income")
    elif dividend_weight > 70:
        print(f"   💡 Suggestion: Well-positioned for income, consider some growth stocks for balance")
    else:
        print(f"   ✅ Good balance between dividend income and growth potential")
    
    # Income growth potential
    print(f"\n🌱 INCOME GROWTH POTENTIAL:")
    growth_stocks_with_divs = []
    for d in dividend_stocks:
        if d['symbol'] in ['MSFT', 'AAPL', 'GOOGL', 'NVDA']:  # Tech stocks that also pay dividends
            growth_stocks_with_divs.append(d['symbol'])
    
    if growth_stocks_with_divs:
        print(f"   🚀 Growth Stocks with Dividends: {', '.join(growth_stocks_with_divs)}")
        print(f"      → Potential for both dividend growth and capital appreciation")
    
    # Income reliability assessment
    etf_dividend_symbols = [d['symbol'] for d in dividend_stocks if d['symbol'] in ['VOO', 'QQQ', 'VYM', 'VHT', 'DLR']]
    if etf_dividend_symbols:
        print(f"   🛡️  Reliable Income Sources (ETFs): {', '.join(etf_dividend_symbols)}")
        print(f"      → Diversified income streams with lower individual stock risk")

def display_advanced_market_sector_analysis(metrics):
    """Display advanced market and sector analysis"""
    
    print("\n" + "🌍" + "="*120)
    print("🌍 ADVANCED MARKET & SECTOR ANALYSIS")
    print("="*120)
    
    sorted_holdings = sorted(
        [h for h in metrics['holdings_summary'] if h['current_value'] > 0], 
        key=lambda x: x['current_value'], 
        reverse=True
    )
    
    total_value = sum(h['current_value'] for h in sorted_holdings)
    
    # Enhanced sector mapping with market context
    sector_mapping = {
        'NVDA': {'sector': 'Technology', 'subsector': 'Semiconductors', 'market_cap': 'Large', 'style': 'Growth'},
        'PLTR': {'sector': 'Technology', 'subsector': 'Software', 'market_cap': 'Large', 'style': 'Growth'},
        'CRWD': {'sector': 'Technology', 'subsector': 'Cybersecurity', 'market_cap': 'Large', 'style': 'Growth'},
        'MSFT': {'sector': 'Technology', 'subsector': 'Software', 'market_cap': 'Large', 'style': 'Growth'},
        'GOOGL': {'sector': 'Technology', 'subsector': 'Internet', 'market_cap': 'Large', 'style': 'Growth'},
        'INTU': {'sector': 'Technology', 'subsector': 'Software', 'market_cap': 'Large', 'style': 'Growth'},
        'TSM': {'sector': 'Technology', 'subsector': 'Semiconductors', 'market_cap': 'Large', 'style': 'Value'},
        'QCOM': {'sector': 'Technology', 'subsector': 'Semiconductors', 'market_cap': 'Large', 'style': 'Value'},
        'AAPL': {'sector': 'Technology', 'subsector': 'Hardware', 'market_cap': 'Large', 'style': 'Growth'},
        'AMZN': {'sector': 'Consumer Discretionary', 'subsector': 'E-commerce', 'market_cap': 'Large', 'style': 'Growth'},
        'TSLA': {'sector': 'Consumer Discretionary', 'subsector': 'Electric Vehicles', 'market_cap': 'Large', 'style': 'Growth'},
        'COST': {'sector': 'Consumer Staples', 'subsector': 'Retail', 'market_cap': 'Large', 'style': 'Value'},
        'OXY': {'sector': 'Energy', 'subsector': 'Oil & Gas', 'market_cap': 'Large', 'style': 'Value'},
        'BAC': {'sector': 'Financials', 'subsector': 'Banking', 'market_cap': 'Large', 'style': 'Value'},
        'BRK.B': {'sector': 'Financials', 'subsector': 'Diversified', 'market_cap': 'Large', 'style': 'Value'},
        # ETFs
        'VOO': {'sector': 'Broad Market', 'subsector': 'S&P 500', 'market_cap': 'Mixed', 'style': 'Blend'},
        'QQQ': {'sector': 'Technology', 'subsector': 'NASDAQ 100', 'market_cap': 'Large', 'style': 'Growth'},
        'VUG': {'sector': 'Broad Market', 'subsector': 'Growth', 'market_cap': 'Large', 'style': 'Growth'},
        'SPY': {'sector': 'Broad Market', 'subsector': 'S&P 500', 'market_cap': 'Mixed', 'style': 'Blend'},
        'VTI': {'sector': 'Broad Market', 'subsector': 'Total Market', 'market_cap': 'Mixed', 'style': 'Blend'},
        'XLY': {'sector': 'Consumer Discretionary', 'subsector': 'Sector ETF', 'market_cap': 'Mixed', 'style': 'Blend'},
        'SMH': {'sector': 'Technology', 'subsector': 'Semiconductors', 'market_cap': 'Mixed', 'style': 'Growth'},
        'VYM': {'sector': 'Broad Market', 'subsector': 'Dividend', 'market_cap': 'Mixed', 'style': 'Value'},
        'SOXX': {'sector': 'Technology', 'subsector': 'Semiconductors', 'market_cap': 'Mixed', 'style': 'Growth'},
        'VHT': {'sector': 'Healthcare', 'subsector': 'Healthcare ETF', 'market_cap': 'Mixed', 'style': 'Blend'},
        'DLR': {'sector': 'Real Estate', 'subsector': 'REITs', 'market_cap': 'Large', 'style': 'Value'}
    }
    
    # Sector allocation analysis
    sector_allocation = defaultdict(float)
    style_allocation = defaultdict(float)
    market_cap_allocation = defaultdict(float)
    
    for holding in sorted_holdings:
        symbol = holding['symbol']
        weight = (holding['current_value'] / total_value * 100)
        
        if symbol in sector_mapping:
            mapping = sector_mapping[symbol]
            sector_allocation[mapping['sector']] += weight
            style_allocation[mapping['style']] += weight
            market_cap_allocation[mapping['market_cap']] += weight
    
    print(f"\n🏢 COMPREHENSIVE SECTOR ALLOCATION:")
    print("-"*120)
    print(f"{'Sector':<25} {'Allocation %':<15} {'Market Context':<30} {'Outlook':<30}")
    print("-"*120)
    
    sector_outlooks = {
        'Technology': 'AI/Cloud growth, rate sensitivity',
        'Consumer Discretionary': 'Economic cycle dependent',
        'Consumer Staples': 'Defensive, stable demand',
        'Energy': 'Commodity cycles, geopolitical',
        'Financials': 'Interest rate sensitive',
        'Healthcare': 'Aging demographics positive',
        'Real Estate': 'Interest rate sensitive',
        'Broad Market': 'Market beta exposure'
    }
    
    for sector, allocation in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True):
        market_context = f"Weight: {allocation:.1f}%"
        outlook = sector_outlooks.get(sector, 'Sector-specific factors')
        
        # Risk assessment
        if allocation > 40:
            risk_level = "⚠️ High Concentration"
        elif allocation > 25:
            risk_level = "⚡ Moderate Concentration"
        else:
            risk_level = "✅ Balanced"
        
        print(f"{sector:<25} {allocation:<15.1f} {risk_level:<30} {outlook:<30}")
    
    # Style analysis
    print(f"\n📊 INVESTMENT STYLE ALLOCATION:")
    print("-"*120)
    for style, allocation in sorted(style_allocation.items(), key=lambda x: x[1], reverse=True):
        style_context = {
            'Growth': 'Higher valuation, earnings growth focus',
            'Value': 'Lower valuation, potential turnaround',
            'Blend': 'Balanced growth and value characteristics'
        }
        print(f"   • {style}: {allocation:.1f}% - {style_context.get(style, '')}")
    
    # Market cap analysis
    print(f"\n💹 MARKET CAPITALIZATION EXPOSURE:")
    print("-"*120)
    for cap, allocation in sorted(market_cap_allocation.items(), key=lambda x: x[1], reverse=True):
        cap_context = {
            'Large': 'Established companies, lower volatility',
            'Mixed': 'Diversified market cap exposure',
            'Mid': 'Growth potential, moderate risk',
            'Small': 'High growth potential, higher risk'
        }
        print(f"   • {cap} Cap: {allocation:.1f}% - {cap_context.get(cap, '')}")
    
    # Market correlation analysis
    print(f"\n🔗 MARKET CORRELATION & BETA ANALYSIS:")
    print("-"*120)
    
    portfolio_beta = metrics.get('portfolio_beta', 1.0)
    high_beta_stocks = []
    low_beta_stocks = []
    
    for holding in sorted_holdings[:10]:  # Top 10 holdings
        symbol = holding['symbol']
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            beta = info.get('beta', None)
            
            if beta:
                if beta > 1.2:
                    high_beta_stocks.append((symbol, beta))
                elif beta < 0.8:
                    low_beta_stocks.append((symbol, beta))
        except:
            continue
    
    print(f"   📊 Portfolio Beta: {portfolio_beta:.2f}")
    
    if high_beta_stocks:
        print(f"   ⚡ High Beta Stocks (>1.2): {', '.join([f'{s}({b:.2f})' for s, b in high_beta_stocks])}")
        print(f"      → More volatile than market, higher risk/reward")
    
    if low_beta_stocks:
        print(f"   🛡️ Low Beta Stocks (<0.8): {', '.join([f'{s}({b:.2f})' for s, b in low_beta_stocks])}")
        print(f"      → Less volatile than market, defensive characteristics")
    
    # Economic sensitivity analysis
    print(f"\n🌊 ECONOMIC CYCLE SENSITIVITY:")
    print("-"*120)
    
    cyclical_exposure = sector_allocation.get('Technology', 0) + sector_allocation.get('Consumer Discretionary', 0)
    defensive_exposure = sector_allocation.get('Consumer Staples', 0) + sector_allocation.get('Healthcare', 0) + sector_allocation.get('Broad Market', 0)
    
    print(f"   📈 Cyclical Exposure: {cyclical_exposure:.1f}% (Tech + Consumer Discretionary)")
    print(f"   🛡️ Defensive Exposure: {defensive_exposure:.1f}% (Staples + Healthcare + Broad Market)")
    
    if cyclical_exposure > 50:
        print(f"   ⚠️ High economic sensitivity - consider defensive positions for downturns")
    elif defensive_exposure > 60:
        print(f"   🛡️ Conservative positioning - consider cyclical exposure for growth")
    else:
        print(f"   ✅ Balanced economic cycle exposure")

def display_portfolio_optimization_analysis(metrics):
    """Display comprehensive portfolio optimization analysis"""
    
    print("\n" + "⚙️" + "="*120)
    print("⚙️ PORTFOLIO OPTIMIZATION & STRATEGIC RECOMMENDATIONS")
    print("="*120)
    
    sorted_holdings = sorted(
        [h for h in metrics['holdings_summary'] if h['current_value'] > 0], 
        key=lambda x: x['current_value'], 
        reverse=True
    )
    
    total_value = sum(h['current_value'] for h in sorted_holdings)
    
    print(f"\n🎯 PORTFOLIO OPTIMIZATION ANALYSIS:")
    print("-"*120)
    
    # Concentration analysis
    top_5_value = sum(h['current_value'] for h in sorted_holdings[:5])
    concentration = (top_5_value / total_value * 100)
    
    print(f"   📊 CONCENTRATION METRICS:")
    print(f"      • Top 5 Holdings: {concentration:.1f}% of portfolio")
    print(f"      • Number of Holdings: {len(sorted_holdings)}")
    print(f"      • Effective Number of Holdings: {1/sum((h['current_value']/total_value)**2 for h in sorted_holdings):.1f}")
    
    # Rebalancing opportunities
    print(f"\n🔄 REBALANCING OPPORTUNITIES:")
    print("-"*120)
    
    # Identify overweight/underweight positions
    overweight_positions = []
    underweight_opportunities = []
    
    for holding in sorted_holdings:
        weight = (holding['current_value'] / total_value * 100)
        symbol = holding['symbol']
        
        if weight > 15:  # Significantly overweight
            overweight_positions.append((symbol, weight))
        elif weight < 1 and holding['current_value'] > 1000:  # Small but meaningful positions
            underweight_opportunities.append((symbol, weight))
    
    if overweight_positions:
        print(f"   ⚠️ OVERWEIGHT POSITIONS (>15%):")
        for symbol, weight in overweight_positions:
            print(f"      • {symbol}: {weight:.1f}% - Consider taking profits or trimming")
    
    if underweight_opportunities:
        print(f"   📈 UNDERWEIGHT OPPORTUNITIES (<1% but meaningful):")
        for symbol, weight in underweight_opportunities:
            print(f"      • {symbol}: {weight:.1f}% - Consider increasing if conviction is high")
    
    # Diversification recommendations
    print(f"\n🌍 DIVERSIFICATION RECOMMENDATIONS:")
    print("-"*120)
    
    # Sector diversification
    sector_allocation = defaultdict(float)
    for holding in sorted_holdings:
        symbol = holding['symbol']
        weight = (holding['current_value'] / total_value * 100)
        
        # Simplified sector mapping
        if symbol in ['NVDA', 'PLTR', 'CRWD', 'MSFT', 'GOOGL', 'INTU', 'TSM', 'QCOM', 'AAPL', 'QQQ', 'SMH', 'SOXX']:
            sector_allocation['Technology'] += weight
        elif symbol in ['VOO', 'SPY', 'VTI', 'VUG']:
            sector_allocation['Broad Market'] += weight
        elif symbol in ['AMZN', 'TSLA', 'XLY']:
            sector_allocation['Consumer Discretionary'] += weight
        elif symbol in ['COST']:
            sector_allocation['Consumer Staples'] += weight
        elif symbol in ['OXY']:
            sector_allocation['Energy'] += weight
        elif symbol in ['BAC', 'BRK.B']:
            sector_allocation['Financials'] += weight
        elif symbol in ['VHT']:
            sector_allocation['Healthcare'] += weight
        elif symbol in ['DLR']:
            sector_allocation['Real Estate'] += weight
        elif symbol in ['VYM']:
            sector_allocation['Dividend Focused'] += weight
    
    missing_sectors = []
    overweight_sectors = []
    
    for sector, weight in sector_allocation.items():
        if weight > 35:
            overweight_sectors.append((sector, weight))
    
    # Check for missing major sectors
    major_sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer Staples', 'Energy', 'Industrials', 'Materials', 'Utilities']
    current_sectors = set(sector_allocation.keys())
    
    for sector in major_sectors:
        if sector not in current_sectors or sector_allocation[sector] < 2:
            missing_sectors.append(sector)
    
    if missing_sectors:
        print(f"   🎯 MISSING SECTOR EXPOSURE:")
        for sector in missing_sectors[:5]:  # Top 5 missing
            sector_suggestions = {
                'Healthcare': 'Consider VHT, JNJ, UNH for healthcare exposure',
                'Industrials': 'Consider IYJ, CAT, GE for industrial exposure', 
                'Materials': 'Consider XLB, FCX, NEM for materials exposure',
                'Utilities': 'Consider XLU, NEE, DUK for utility exposure',
                'Financials': 'Consider XLF, JPM, BAC for financial exposure'
            }
            suggestion = sector_suggestions.get(sector, f'Consider {sector} sector ETF or individual stocks')
            print(f"      • {sector}: {suggestion}")
    
    if overweight_sectors:
        print(f"   ⚠️ OVERWEIGHT SECTORS (>35%):")
        for sector, weight in overweight_sectors:
            print(f"      • {sector}: {weight:.1f}% - Consider reducing concentration")
    
    # Geographic diversification
    print(f"\n🌍 GEOGRAPHIC DIVERSIFICATION:")
    us_weight = sum((h['current_value'] / total_value * 100) for h in sorted_holdings if h['symbol'] not in ['TSM'])
    international_weight = 100 - us_weight
    
    print(f"   • US Exposure: ~{us_weight:.1f}%")
    print(f"   • International Exposure: ~{international_weight:.1f}%")
    
    if international_weight < 20:
        print(f"   💡 Consider adding international exposure through VXUS, VEA, VWO, or international ETFs")
    
    # Risk optimization
    print(f"\n⚖️ RISK OPTIMIZATION RECOMMENDATIONS:")
    print("-"*120)
    
    portfolio_volatility = metrics.get('portfolio_volatility', 0)
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    
    print(f"   📊 Current Risk Metrics:")
    print(f"      • Portfolio Volatility: {portfolio_volatility:.1f}%")
    print(f"      • Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"      • Risk Level: {'High' if portfolio_volatility > 25 else 'Medium' if portfolio_volatility > 15 else 'Low'}")
    
    # Risk optimization suggestions
    if portfolio_volatility > 20:
        print(f"   🛡️ TO REDUCE RISK:")
        print(f"      • Increase allocation to broad market ETFs (VOO, VTI)")
        print(f"      • Add defensive sectors (utilities, consumer staples)")
        print(f"      • Consider bonds or fixed income allocation")
        print(f"      • Reduce concentration in high-volatility growth stocks")
    
    if sharpe_ratio < 0.5:
        print(f"   📈 TO IMPROVE RISK-ADJUSTED RETURNS:")
        print(f"      • Rebalance toward higher-performing, lower-risk positions")
        print(f"      • Consider reducing underperforming positions")
        print(f"      • Add dividend-paying stocks for consistent income")
    
    # Tax optimization
    print(f"\n💰 TAX OPTIMIZATION OPPORTUNITIES:")
    print("-"*120)
    
    # Identify tax loss harvesting opportunities
    loss_positions = [(h['symbol'], h.get('unrealized_gain', 0)) for h in sorted_holdings if h.get('unrealized_gain', 0) < -1000]
    gain_positions = [(h['symbol'], h.get('unrealized_gain', 0)) for h in sorted_holdings if h.get('unrealized_gain', 0) > 5000]
    
    if loss_positions:
        print(f"   📉 TAX LOSS HARVESTING OPPORTUNITIES:")
        for symbol, loss in sorted(loss_positions, key=lambda x: x[1])[:3]:
            print(f"      • {symbol}: ${loss:,.0f} unrealized loss - Consider harvesting for tax benefits")
    
    if gain_positions:
        print(f"   📈 LARGE UNREALIZED GAINS (Tax Planning):")
        for symbol, gain in sorted(gain_positions, key=lambda x: x[1], reverse=True)[:3]:
            print(f"      • {symbol}: ${gain:,.0f} unrealized gain - Consider tax implications of selling")
    
    # Final optimization summary
    print(f"\n🎯 OPTIMIZATION PRIORITY RECOMMENDATIONS:")
    print("-"*120)
    print(f"   1. 🔄 IMMEDIATE ACTIONS:")
    
    if concentration > 60:
        print(f"      • Reduce concentration risk - top 5 holdings are {concentration:.1f}% of portfolio")
    
    if len(missing_sectors) > 3:
        print(f"      • Add sector diversification - missing exposure to {len(missing_sectors)} major sectors")
    
    if international_weight < 15:
        print(f"      • Add international diversification - currently only {international_weight:.1f}%")
    
    print(f"\n   2. 📈 MEDIUM-TERM GOALS:")
    print(f"      • Target 15-25% international exposure")
    print(f"      • Maintain 5-10% in each major sector")
    print(f"      • Keep individual positions under 15% of portfolio")
    print(f"      • Aim for Sharpe ratio > 0.7")
    
    print(f"\n   3. 🎯 LONG-TERM STRATEGY:")
    print(f"      • Regular rebalancing (quarterly or semi-annually)")
    print(f"      • Tax-efficient harvesting and positioning")
    print(f"      • Consider lifecycle changes and risk tolerance")
    print(f"      • Monitor and adjust based on market conditions")

def save_analysis_to_pdf(content, filename="portfolio_analysis.pdf"):
    """Save the analysis content to a PDF with markdown-like formatting"""
    
    print(f"\n💾 Generating PDF report: {filename}")
    
    # Create PDF document
    doc = SimpleDocTemplate(filename, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles for different elements
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        alignment=TA_LEFT
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        alignment=TA_LEFT
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
        alignment=TA_LEFT,
        fontName='Courier'  # Monospace for better formatting
    )
    
    # Split content into lines and process
    lines = content.split('\n')
    
    for line in lines:
        if not line.strip():
            story.append(Spacer(1, 6))
            continue
            
        # Remove ANSI color codes and special characters that might cause issues
        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        clean_line = clean_line.replace('═', '=').replace('─', '-')
        
        # Determine line type and apply appropriate style
        if clean_line.startswith('🚀') or clean_line.startswith('🎯') or 'COMPREHENSIVE' in clean_line:
            # Main titles
            text_content = clean_line.strip('🚀🎯 =')
            story.append(Paragraph(text_content, title_style))
        elif any(clean_line.startswith(prefix) for prefix in ['📊', '🔍', '💰', '⚠️', '💡', '🌍', '🎯']):
            # Section headers
            story.append(Paragraph(clean_line, heading_style))
        elif clean_line.strip().startswith('=') and len(clean_line.strip()) > 50:
            # Separator lines - add space instead
            story.append(Spacer(1, 8))
        else:
            # Regular content
            if clean_line.strip():
                story.append(Paragraph(clean_line, normal_style))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"✅ PDF report saved successfully: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False

def save_analysis_to_html(content, metrics, analyzer, filename="portfolio_analysis.html"):
    """Save the analysis content and interactive charts to an HTML file"""

    print(f"\n💾 Generating HTML report: {filename}")

    try:
        holdings_df = pd.DataFrame(metrics.get('holdings_summary', []))
        if not holdings_df.empty:
            table_html = holdings_df[
                ['symbol', 'quantity', 'current_value', 'unrealized_gain_pct']
            ].to_html(index=False, float_format=lambda x: f"{x:,.2f}")
        else:
            table_html = "<p>No holdings data available.</p>"

        sectors = list(metrics.get('sector_allocation', {}).keys())
        values = list(metrics.get('sector_allocation', {}).values())
        charts = []
        if sectors and values:
            fig = px.pie(names=sectors, values=values, title="Sector Allocation")
            charts.append(pio.to_html(fig, include_plotlyjs='cdn', full_html=False))

        # Portfolio performance over time vs benchmark
        portfolio_series = []
        for symbol, holding in analyzer.holdings.items():
            hist = analyzer.historical_data.get(symbol, pd.DataFrame())
            if hist.empty or 'Close' not in hist:
                continue
            df = hist[['Close']].rename(columns={'Close': symbol}) * holding['quantity']
            portfolio_series.append(df)

        if portfolio_series:
            combined = pd.concat(portfolio_series, axis=1).fillna(method='ffill')
            combined['Portfolio'] = combined.sum(axis=1)
            if hasattr(analyzer, 'benchmark_data') and not getattr(analyzer, 'benchmark_data', pd.DataFrame()).empty:
                bench = analyzer.benchmark_data[['Close']].rename(columns={'Close': 'Benchmark'})
                combined = combined.join(bench, how='inner')
            line_df = combined[['Portfolio'] + (["Benchmark"] if 'Benchmark' in combined.columns else [])].dropna().reset_index()
            line_df.rename(columns={'index': 'Date'}, inplace=True)
            fig_line = px.line(line_df, x='Date', y=line_df.columns[1:], title='Portfolio Performance')
            charts.append(pio.to_html(fig_line, include_plotlyjs='cdn', full_html=False))

        # Risk-return scatter plot
        risk_data = []
        for symbol, hist in analyzer.historical_data.items():
            if hist.empty or 'Close' not in hist:
                continue
            returns = hist['Close'].pct_change().dropna()
            if returns.empty:
                continue
            volatility = returns.std() * np.sqrt(252)
            cumulative = (1 + returns).prod() - 1
            risk_data.append({'Symbol': symbol, 'Volatility': volatility * 100, 'Return': cumulative * 100})

        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            fig_scatter = px.scatter(risk_df, x='Volatility', y='Return', text='Symbol', title='Risk vs Return')
            charts.append(pio.to_html(fig_scatter, include_plotlyjs='cdn', full_html=False))

        # Treemap of portfolio composition
        if not holdings_df.empty and 'sector' in holdings_df.columns:
            fig_tree = px.treemap(holdings_df, path=['sector', 'symbol'], values='current_value', title='Portfolio Composition')
            charts.append(pio.to_html(fig_tree, include_plotlyjs='cdn', full_html=False))

        charts_html = "\n".join(charts) if charts else "<p>No chart data available.</p>"

        html_content = f"""
<html>
<head>
    <meta charset='utf-8'>
    <title>Portfolio Analysis Report</title>
    <style>body{{font-family:Arial, sans-serif;}}</style>
</head>
<body>
    <h1>Portfolio Analysis Report</h1>
    <pre>{html.escape(content)}</pre>
    <h2>Holdings Summary</h2>
    {table_html}
    <h2>Interactive Charts</h2>
    {charts_html}
</body>
</html>
"""

        # Write using UTF-8 to handle emojis and other unicode characters
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ HTML report saved successfully: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error generating HTML: {e}")
        return False

def capture_analysis_output(return_metrics: bool = False, return_analyzer: bool = False):
    """Capture all analysis output to a string.

    If ``return_metrics`` is True, return the metrics dictionary as well.
    If ``return_analyzer`` is True, return the analyzer instance.
    """


    
    output_buffer = io.StringIO()
    
    try:
        # Redirect stdout to capture all print statements
        with redirect_stdout(output_buffer):
            print("🚀" + "="*120)
            print("🎯 COMPREHENSIVE PORTFOLIO ANALYSIS SUITE")
            print("👤 Author: Divakar")
            print("📅 Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("="*120)
            
            print("📂 Loading portfolio data...")
            analyzer = EnhancedPortfolioAnalyzer("portfolio.csv")
            analyzer.load_data()
            
            print("🧮 Calculating holdings...")
            analyzer.calculate_holdings()
            
            print("📡 Fetching market data...")
            analyzer.get_market_data()
            
            print("📊 Computing portfolio metrics...")
            metrics = analyzer.calculate_portfolio_metrics()
            
            if not metrics:
                raise Exception("Failed to calculate portfolio metrics")
            
            print("\n" + "✅" + "="*60)
            print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
            print("="*60)
            
            # Display main metrics
            print(f"📊 Total Return: ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.1f}%)")
            print(f"💰 Current Value: ${metrics['current_value']:,.2f}")
            print(f"💵 Total Invested: ${metrics['total_invested']:,.2f}")
            print(f"📈 Unrealized Gains: ${metrics['unrealized_gains']:,.2f}")
            print(f"💸 Dividend Income: ${metrics['dividend_income']:,.2f}")
            print(f"🎯 Diversification Score: {metrics['diversification_score']:.0f}/100")
            print(f"📊 Portfolio Beta: {metrics['portfolio_beta']:.2f}")
            print(f"⚡ Portfolio Volatility: {metrics['portfolio_volatility']:.1f}%")
            print(f"🏆 Best Performer: {metrics['best_performer']['symbol']} ({metrics['best_performer']['return_pct']:.1f}%)")
            print(f"⚠️  Worst Performer: {metrics['worst_performer']['symbol']} ({metrics['worst_performer']['return_pct']:.1f}%)")
            
            print(f"\n📋 Analysis Details:")
            print(f"   🔍 Individual Stock Profiles: {len([h for h in metrics['holdings_summary'] if h['current_value'] > 1])} detailed analyses")
            print(f"   📊 Portfolio Contribution Analysis: Complete")
            print(f"   💡 Investment Recommendations: Available for all holdings")
            print(f"   📈 Total Holdings Analyzed: {metrics['num_holdings']} active positions")
            print(f"   🌐 Sectors Covered: {metrics['num_sectors']} different sectors")
            
            # Display all analysis sections
            display_detailed_holdings_table(metrics)
            display_strategic_analysis(metrics)
            display_sector_analysis(metrics)
            display_risk_analysis(metrics)
            display_investment_recommendations(metrics)
            display_fundamental_analysis(metrics)
            display_performance_attribution_analysis(metrics)
            display_income_dividend_analysis(metrics)
            display_advanced_market_sector_analysis(metrics)
            display_portfolio_optimization_analysis(metrics)
            
            print("\n" + "🎉" + "="*120)
            print("🎉 COMPLETE ADVANCED PORTFOLIO ANALYSIS FINISHED!")
            print("="*120)
            print("📊 Comprehensive analysis including fundamentals, attribution, income, and optimization")
            print("🚀 All analysis displayed in rich console format")
            print("="*120)
            
            # Add disclaimer
            print("\n" + "⚠️" + "="*120)
            print("⚠️ IMPORTANT DISCLAIMER")
            print("="*120)
            print("📋 This analysis is for informational purposes only and should not be considered as")
            print("   investment advice. Past performance does not guarantee future results.")
            print("💡 Key Points:")
            print("   • All data is sourced from public APIs and may contain inaccuracies")
            print("   • Market conditions can change rapidly, affecting portfolio performance")
            print("   • This tool does not account for individual risk tolerance or financial goals")
            print("   • Consider consulting with a qualified financial advisor before making investment decisions")
            print("   • The author (Divakar) is not responsible for any investment losses")
            print("📞 Always do your own research and consider multiple sources before investing")
            print("="*120)
        
        # Get the captured output
        content = output_buffer.getvalue()
        results = [content]
        if return_metrics:
            results.append(metrics)
        if return_analyzer:
            results.append(analyzer)
        if len(results) == 1:
            return results[0]
        return tuple(results)
        
    except Exception as e:
        return f"❌ Error capturing analysis: {e}"
    finally:
        output_buffer.close()

def main():
    """Main function with command line argument parsing"""
    
    parser = argparse.ArgumentParser(description='Portfolio Analysis Tool')
    parser.add_argument('--save', action='store_true',
                       help='Save analysis to PDF file in addition to console output')
    parser.add_argument('--html', action='store_true',
                       help='Save analysis to interactive HTML report')
    
    args = parser.parse_args()
    
    if args.save:
        # Capture output and save to PDF
        print("🔄 Running analysis and capturing output for PDF...")
        content = capture_analysis_output()
        
        # Also display to console
        print(content)
        
        # Save to PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_analysis_{timestamp}.pdf"
        success = save_analysis_to_pdf(content, filename)
        
        if success:
            print(f"\n📁 Analysis saved to: {filename}")

    if args.html:
        print("🔄 Running analysis and capturing output for HTML...")
        content, metrics, analyzer = capture_analysis_output(return_metrics=True, return_analyzer=True)

        print(content)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"portfolio_analysis_{timestamp}.html"
        success_html = save_analysis_to_html(content, metrics, analyzer, html_filename)

        if success_html:
            print(f"\n📁 Analysis saved to: {html_filename}")

    if not (args.save or args.html):
        # Just run normal console analysis
        display_comprehensive_portfolio_analysis()

if __name__ == "__main__":
    main()
