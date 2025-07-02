# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Planned: Cryptocurrency portfolio analysis
- Planned: Real-time portfolio tracking
- Planned: Web interface for easier use
- Planned: International markets support

## [1.0.0] - 2025-07-02

### Added

- Initial release of Robinhood Portfolio Analyzer
- Portfolio performance analysis with comprehensive metrics
- Risk assessment calculations (volatility, VaR, maximum drawdown)
- Dividend tracking and reinvestment analysis
- Morningstar-style stock quality and valuation ratings
- Interactive visualizations using Plotly
- PDF report generation with ReportLab
- Sample data generator for testing
- Support for CSV portfolio data import
- Command-line interface with customizable options
- Comprehensive documentation and setup guides

### Features

- Calculate alpha, beta, Sharpe ratio, and other performance metrics
- Generate risk-return scatter plots
- Create asset allocation pie charts and treemaps
- Compare portfolio performance against market benchmarks
- Export analysis results in multiple formats (PDF, HTML, CSV, PNG/SVG)
- Process dividend payments and reinvestments
- Handle various transaction types from Robinhood exports

### Dependencies

- pandas >= 1.5.0 for data manipulation
- numpy >= 1.21.0 for numerical computations
- matplotlib >= 3.5.0 for static plotting
- seaborn >= 0.11.0 for statistical visualizations
- yfinance >= 0.2.0 for stock market data
- reportlab >= 3.6.0 for PDF generation
- plotly >= 5.0.0 for interactive charts
- scipy >= 1.9.0 for statistical analysis
- kaleido >= 0.2.1 for static image export
