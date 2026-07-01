"""
Customer (merchant) lifetime value.

A payments company does not earn a merchant's full transaction volume, it
earns a cut of it. All CLV figures here are PLATFORM REVENUE, meaning
merchant GMV multiplied by an assumed net take rate, not raw GMV. This
matters: a merchant processing $500k a year is not worth $500k to the
platform, they are worth roughly $500k x take rate.

Assumptions (documented so they can be argued with, which is the point):
- Take rate: 1.8% of successful transaction volume. This approximates a
  blended net take rate after interchange and scheme fees for a mixed
  card-present / card-not-present book. A real business would plug in its
  own actual blended rate here.
- Currently churned: a merchant with no successful transaction in the last
  60 days as of the snapshot date. 60 days of silence from a merchant that
  was transacting regularly is a strong, standard churn signal.
- Expected remaining lifetime for an active merchant = 1 / predicted monthly
  churn probability. This is the standard textbook relationship between a
  constant monthly churn hazard and expected customer lifetime (it assumes
  the hazard stays constant, which is a simplification, not a guarantee).
  Capped at 60 months so a near-zero churn probability does not produce an
  absurd lifetime.
- A merchant already flagged as currently churned gets zero predicted future
  value. Projecting a dead merchant's historic run-rate forward would
  overstate their worth; the honest estimate for a merchant that is gone is
  what they already generated, not more of the same.

Total CLV = historic platform revenue + predicted future platform revenue.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TAKE_RATE = 0.018
CHURN_RECENCY_THRESHOLD_DAYS = 60
MAX_EXPECTED_LIFETIME_MONTHS = 60
DAYS_PER_MONTH = 30.44

NUMERIC_FEATURES = ["frequency", "monetary", "avg_ticket_size", "weekly_txn_rate", "decline_rate", "tenure_months"]
CATEGORICAL_FEATURES = ["industry"]


def add_revenue_and_tenure(rfm: pd.DataFrame, merchants: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    df = rfm.merge(
        merchants[["merchant_id", "avg_ticket_size", "weekly_txn_rate", "decline_rate"]],
        on="merchant_id", how="left",
    )

    df["historic_platform_revenue"] = df["monetary"] * TAKE_RATE
    df["tenure_days"] = (snapshot_date - df["signup_date"]).dt.days
    df["tenure_months"] = (df["tenure_days"] / DAYS_PER_MONTH).clip(lower=1)
    df["monthly_revenue_rate"] = df["historic_platform_revenue"] / df["tenure_months"]
    df["is_currently_churned"] = df["recency_days"] > CHURN_RECENCY_THRESHOLD_DAYS

    return df


def train_churn_model(df: pd.DataFrame):
    """
    Logistic regression predicting whether a merchant is currently churned,
    using behavioral features only. Recency is deliberately excluded: it is
    what defines the label, so including it would be circular, not predictive.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_currently_churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    model.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, test_auc


def predict_clv(df: pd.DataFrame, model) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    df = df.copy()
    df["churn_probability"] = model.predict_proba(X)[:, 1]

    # floor prevents division by ~0 from producing an unrealistic lifetime
    monthly_churn = df["churn_probability"].clip(lower=1 / MAX_EXPECTED_LIFETIME_MONTHS)
    df["expected_remaining_months"] = (1 / monthly_churn).clip(upper=MAX_EXPECTED_LIFETIME_MONTHS)

    df["predicted_future_revenue"] = np.where(
        df["is_currently_churned"],
        0.0,
        df["monthly_revenue_rate"] * df["expected_remaining_months"],
    )

    df["total_clv"] = df["historic_platform_revenue"] + df["predicted_future_revenue"]

    return df


def build_clv_table(rfm: pd.DataFrame, merchants: pd.DataFrame, transactions: pd.DataFrame):
    snapshot_date = transactions["transaction_date"].max() + pd.Timedelta(days=1)
    df = add_revenue_and_tenure(rfm, merchants, snapshot_date)
    model, test_auc = train_churn_model(df)
    df = predict_clv(df, model)
    return df, test_auc
