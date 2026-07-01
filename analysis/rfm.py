"""
RFM segmentation for payment merchants.

RFM stands for Recency, Frequency, Monetary value. It is a standard way to
group customers (here, merchants) by how recently they transacted, how often
they transact, and how much money they move. The method is old and well
understood, which is exactly why it is a good baseline: it is easy to defend,
easy to explain to a non-technical stakeholder, and gives a segment label
that a product or risk team can act on directly.

Definitions used here:
- Recency:  days between a merchant's last SUCCESSFUL transaction and the
            snapshot date (the day after the last transaction in the data).
- Frequency: count of successful transactions for that merchant.
- Monetary:  total dollar amount of successful transactions for that merchant.

Only successful transactions count toward F and M. A failed charge is not
revenue and should not inflate a merchant's value score. Failure behavior is
analyzed separately in insights.py so it does not get buried inside RFM.

Merchants with zero successful transactions get the worst possible recency
(days since signup) and F = M = 0. That is not an edge case to special-case
away, it is a real, meaningful state: a merchant who never converted.
"""

import pandas as pd


def compute_rfm_table(merchants: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    successful = transactions[transactions["status"] == "Success"].copy()

    snapshot_date = transactions["transaction_date"].max() + pd.Timedelta(days=1)

    agg = successful.groupby("merchant_id").agg(
        last_transaction_date=("transaction_date", "max"),
        frequency=("transaction_id", "count"),
        monetary=("amount", "sum"),
    ).reset_index()

    rfm = merchants[["merchant_id", "industry", "signup_date"]].merge(agg, on="merchant_id", how="left")

    # Merchants with no successful transaction: treat "last activity" as their
    # signup date, so recency reflects how long they have gone without ever
    # converting rather than producing a null.
    rfm["last_transaction_date"] = rfm["last_transaction_date"].fillna(rfm["signup_date"])
    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    rfm["monetary"] = rfm["monetary"].fillna(0.0)

    rfm["recency_days"] = (snapshot_date - rfm["last_transaction_date"]).dt.days

    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Splits R, F, M into quintiles (1 low - 5 high). Recency is inverted so a
    LOW day count (transacted recently) gets a HIGH score, matching the
    convention that a higher RFM score always means a more valuable merchant.

    qcut can collapse buckets when a column has many repeated values (common
    with frequency, since lots of merchants land on the same integer count).
    duplicates="drop" keeps the function from crashing on that; it just means
    fewer than 5 distinct buckets for that column, which is expected behavior,
    not a bug.
    """
    scored = rfm.copy()

    scored["R_score"] = pd.qcut(
        scored["recency_days"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"
    ).astype(int)
    scored["F_score"] = pd.qcut(
        scored["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype(int)
    scored["M_score"] = pd.qcut(
        scored["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype(int)

    scored["FM_score"] = (scored["F_score"] + scored["M_score"]) / 2

    return scored


SEGMENT_DEFINITIONS = {
    "Champions": "Transacted recently, transact often, and process high volume. The merchants a payments company builds its revenue around.",
    "Loyal": "Consistent, reliable transaction history and solid volume, just short of the top tier. The dependable core of the book.",
    "New": "Signed up and started transacting recently, but have not built up volume yet. Too early to tell if they become Champions or churn.",
    "Needs Attention": "Middling on recency, frequency, and volume. Not at risk yet, but trending flat rather than growing.",
    "At Risk": "Used to transact frequently and at real volume, but have gone quiet recently. High value still on the table if they come back.",
    "Dormant": "Low volume, low frequency, and long gone. Little activity to build on.",
}


def assign_segment(row) -> str:
    r, fm = row["R_score"], row["FM_score"]

    if r >= 4 and fm >= 4:
        return "Champions"
    if r >= 3 and fm >= 3.5:
        return "Loyal"
    if r >= 4 and fm <= 2.5:
        return "New"
    if r <= 2 and fm >= 3.5:
        return "At Risk"
    if r <= 2 and fm <= 2.5:
        return "Dormant"
    return "Needs Attention"


def build_rfm_segments(merchants: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    rfm = compute_rfm_table(merchants, transactions)
    scored = score_rfm(rfm)
    scored["segment"] = scored.apply(assign_segment, axis=1)
    scored["segment_description"] = scored["segment"].map(SEGMENT_DEFINITIONS)
    return scored
