# Robinhood Portfolio Analyzer - Development Setup

## Development Environment Setup

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### Setting up Development Environment

1. **Clone the repository**:

```bash
git clone https://github.com/rexdivakar/robinhood_portfolio_analyzer.git
cd robinhood_portfolio_analyzer
```

2. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If you create this for dev dependencies
```

### Code Quality Tools

We recommend using the following tools for development:

- **Black** for code formatting: `pip install black`
- **Flake8** for linting: `pip install flake8`
- **Pytest** for testing: `pip install pytest`

### Running Tests

```bash
python -m pytest tests/ -v
```

### Code Formatting

```bash
black main.py generate_data.py
```

### Linting

```bash
flake8 main.py generate_data.py
```

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
├── DEVELOPMENT.md             # This file
├── package.json               # Project metadata
└── tests/                     # Test files (if created)
    └── test_main.py
```

## Contributing Guidelines

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests and ensure code quality
5. Commit your changes: `git commit -am 'Add some feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## Release Process

1. Update version in `package.json`
2. Update `CHANGELOG.md`
3. Create a new release tag
4. Update documentation if needed
