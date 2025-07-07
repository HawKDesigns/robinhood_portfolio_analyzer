# Robinhood Portfolio Analyzer

A Python-based tool for analyzing and visualizing your investment portfolio data. This tool helps you gain insights into your investments through detailed analysis and generates comprehensive reports.

## Features

### Core Analysis

- **Portfolio Performance Analysis**: Comprehensive performance metrics including total returns, alpha, beta, Sharpe ratio
- **Risk Assessment**: Calculate volatility, Value at Risk (VaR), maximum drawdown, and risk ratings
- **Dividend Tracking**: Track dividend payments and reinvestments
- **Stock Quality Ratings**: Morningstar-style quality and valuation assessments

### Visualizations

- **Interactive Charts**: Dynamic plots using Plotly for portfolio performance over time
- **Risk-Return Analysis**: Scatter plots showing risk vs return characteristics
- **Asset Allocation**: Pie charts and treemaps for portfolio composition
- **Performance Comparison**: Benchmark comparisons with market indices

### Reporting

- **PDF Reports**: Professional-grade reports with charts and analysis
- **Export Options**: Save data and charts in various formats
- **Customizable Templates**: Flexible report layouts and styling

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/rexdivakar/robinhood_portfolio_analyzer.git
    cd robinhood_portfolio_analyzer
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Quick Start

1. **Prepare your portfolio data**: Export your portfolio data from Robinhood in CSV format and save it as `portfolio.csv`

2. **Run the analysis**:

    ```bash
    python main.py
    ```

### CSV Data Format

Your `portfolio.csv` should include the following columns:

- `Activity Date`: Date of the transaction (MM/DD/YYYY)
- `Process Date`: Processing date
- `Settle Date`: Settlement date
- `Instrument`: Stock symbol (e.g., AAPL, MSFT)
- `Description`: Transaction description
- `Trans Code`: Transaction type (CDIV for dividends, Buy for purchases)
- `Quantity`: Number of shares
- `Price`: Price per share
- `Amount`: Total transaction amount

### Sample Data Generation

If you want to test the tool with sample data, use the included data generator:

```bash
python generate_data.py
```

This will create a sample `portfolio.csv` with realistic portfolio data for testing.

### Command Line Options

The tool supports various command line arguments for customization:

```bash
python main.py --help
```

Common examples:

```bash
python main.py --save    # generate PDF report
python main.py --html    # generate interactive HTML report
```

## Output

The analyzer generates:

- **Interactive HTML reports** with dynamic charts
- **PDF reports** with comprehensive analysis
- **CSV exports** of calculated metrics
- **PNG/SVG charts** for presentations

## Dependencies

- pandas (≥1.5.0)
- numpy (≥1.21.0)
- matplotlib (≥3.5.0)
- seaborn (≥0.11.0)
- yfinance (≥0.2.0)
- reportlab (≥3.6.0)
- requests (≥2.28.0)
- plotly (≥5.0.0)
- scipy (≥1.9.0)
- kaleido (≥0.2.1)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For detailed development setup instructions, see [DEVELOPMENT.md](DEVELOPMENT.md).

### Development Setup

1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Make your changes
6. Run tests and formatting checks
7. Submit a pull request

## Project Structure

```
robinhood_portfolio_analyzer/
├── main.py                    # Main application script
├── generate_data.py           # Sample data generator
├── portfolio.csv              # Sample portfolio data
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore patterns
├── DEVELOPMENT.md             # Development setup guide
├── CHANGELOG.md               # Version history and changes
├── config.template.json       # Configuration template
└── package.json               # Project metadata
```

## Configuration

You can customize the analysis by creating a `config.json` file based on the provided template:

```bash
cp config.template.json config.json
```

Then edit `config.json` to adjust settings like:

- Portfolio input file path
- Benchmark index for comparison
- Risk-free rate for calculations
- Output formats and directories
- Chart themes and styling

## Troubleshooting

### Common Issues

1. **ImportError for yfinance or other packages**:

   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Empty or malformed CSV file**:
   - Ensure your CSV follows the exact format specified above
   - Use the sample data generator to test: `python generate_data.py`

3. **Memory issues with large portfolios**:
   - The tool is optimized for portfolios with up to 10,000 transactions
   - For larger datasets, consider filtering by date range

### Performance Tips

- Use SSD storage for better I/O performance
- Close other memory-intensive applications
- Consider using Python 3.8+ for better performance

## Roadmap

- [ ] Add support for cryptocurrency analysis
- [ ] Implement real-time portfolio tracking
- [ ] Add more benchmark comparison options
- [ ] Create web interface for easier use
- [ ] Add support for international markets

## Changelog

### Version 1.0.0 (Current)

- Initial release with core portfolio analysis features
- PDF and HTML report generation
- Risk metrics and performance calculations
- Interactive visualizations

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/rexdivakar/robinhood_portfolio_analyzer/issues) page
2. Create a new issue with detailed information about your problem
3. Include your Python version and OS information

## Acknowledgments

- Thanks to the maintainers of yfinance for stock data access
- Plotly team for excellent visualization capabilities
- ReportLab for PDF generation functionality
