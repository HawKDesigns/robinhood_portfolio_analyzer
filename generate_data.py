import pandas as pd
import random
from datetime import datetime, timedelta

# Define instruments and CUSIPs
instruments = {
    "VOO": "922908363", "AAPL": "037833100", "SCHD": "808524797", "MSFT": "594918104",
    "XLY": "81369Y407", "TSLA": "88160R101", "VYM": "921946406", "QCOM": "747525103",
    "SPY": "78462F103", "JNJ": "478160104"
}
names = {
    "VOO": "Vanguard S&P 500 ETF", "AAPL": "Apple Inc", "SCHD": "Schwab US Dividend Equity ETF",
    "MSFT": "Microsoft Corporation", "XLY": "Consumer Discretionary Select Sector SPDR Fund",
    "TSLA": "Tesla Inc", "VYM": "Vanguard High Dividend Yield ETF", "QCOM": "Qualcomm",
    "SPY": "SPDR S&P 500 ETF Trust", "JNJ": "Johnson & Johnson"
}
price_ranges = {
    "VOO": (450, 550), "AAPL": (200, 300), "SCHD": (70, 90), "MSFT": (400, 500),
    "XLY": (180, 250), "TSLA": (300, 450), "VYM": (110, 150), "QCOM": (140, 180),
    "SPY": (500, 600), "JNJ": (130, 170)
}

# Generate dates
start_date = datetime(2024, 7, 1)
end_date = datetime(2025, 7, 2)
date_range = (end_date - start_date).days

# Generate 1000 records
records = []
for _ in range(1000):
    activity_date = start_date + timedelta(days=random.randint(0, date_range))
    activity_date_str = activity_date.strftime("%m/%d/%Y")
    
    trans_code = random.choice(["CDIV", "Buy"])
    instrument = random.choice(list(instruments.keys()))
    cusip = instruments[instrument]
    name = names[instrument]
    
    if trans_code == "CDIV":
        record_date = activity_date - timedelta(days=random.randint(3, 7))
        record_date_str = record_date.strftime("%m/%d/%Y")
        process_date = activity_date
        settle_date = activity_date
        shares = round(random.uniform(0.1, 0.5), 6)
        div_rate = round(random.uniform(0.4, 1.2), 4)
        amount = round(shares * div_rate, 2)
        description = f"{name} CUSIP: {cusip} Cash Div: R/D {record_date_str} P/D {activity_date_str} - {shares} shares at {div_rate}"
        quantity = ""
        price = ""
        amount_str = f"${amount:.2f}"
    else:
        process_date = activity_date
        settle_date = activity_date + timedelta(days=random.randint(1, 2))
        quantity = round(random.uniform(0.0001, 0.0015), 6)
        price = round(random.uniform(*price_ranges[instrument]), 2)
        amount = round(quantity * price, 2)
        description = f"{name} CUSIP: {cusip} Dividend Reinvestment"
        quantity = f"{quantity:.6f}"
        price = f"${price:.2f}"
        amount_str = f"(${amount:.2f})"
    
    process_date_str = process_date.strftime("%m/%d/%Y")
    settle_date_str = settle_date.strftime("%m/%d/%Y")
    
    records.append({
        "Activity Date": activity_date_str,
        "Process Date": process_date_str,
        "Settle Date": settle_date_str,
        "Instrument": instrument,
        "Description": description,
        "Trans Code": trans_code,
        "Quantity": quantity,
        "Price": price,
        "Amount": amount_str
    })

# Create DataFrame and save to CSV
df = pd.DataFrame(records)
df.to_csv("portfolio.csv", index=False)
print("Generated 1000 records and saved to portfolio.csv")