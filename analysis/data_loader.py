"""
Loads the merchant and transaction CSVs and does the light cleanup every
downstream analysis needs (typed dates, a couple of derived columns).
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_merchants(data_dir=DATA_DIR):
    path = os.path.join(data_dir, "merchants.csv")
    merchants = pd.read_csv(path, parse_dates=["signup_date"])
    return merchants


def load_transactions(data_dir=DATA_DIR):
    path = os.path.join(data_dir, "transactions.csv")
    transactions = pd.read_csv(path, parse_dates=["transaction_date"])
    return transactions


def load_all(data_dir=DATA_DIR):
    merchants = load_merchants(data_dir)
    transactions = load_transactions(data_dir)
    return merchants, transactions
