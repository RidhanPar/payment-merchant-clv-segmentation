"""
Payment Merchant Segmentation & Lifetime Value Dashboard
RFM segmentation, CLV estimation, and a failure-rate exposure view for a
simulated payments merchant book.
Run: streamlit run app.py
"""

import os
import subprocess

import streamlit as st
import pandas as pd
import plotly.express as px

from analysis.data_loader import load_all
from analysis.rfm import build_rfm_segments, SEGMENT_DEFINITIONS
from analysis.clv import build_clv_table, TAKE_RATE
from analysis.insights import build_segment_summary, generate_recommendations

st.set_page_config(page_title="Merchant CLV & Segmentation", page_icon="💳", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.kpi-card {
    background: #ffffff;
    border: 1px solid #e6e9ee;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    height: 100%;
}
.kpi-label {
    font-size: 12px; font-weight: 600; color: #7c8798;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;
}
.kpi-value { font-size: 26px; font-weight: 700; color: #101b2d; margin-bottom: 4px; line-height: 1.1; }
.kpi-sub { font-size: 12px; font-weight: 500; color: #7c8798; }

.page-header {
    background: linear-gradient(135deg, #0b2545 0%, #0e7c7b 100%);
    border-radius: 12px; padding: 24px 32px; margin-bottom: 24px; color: white;
}
.page-header h1 { font-size: 24px; font-weight: 700; margin: 0; color: white; }
.page-header p { font-size: 13px; margin: 4px 0 0 0; color: rgba(255,255,255,0.8); }

.section-header {
    font-size: 16px; font-weight: 700; color: #101b2d;
    padding-bottom: 8px; border-bottom: 2px solid #0e7c7b; margin-bottom: 16px; margin-top: 8px;
}

.rec-card {
    background: #ffffff; border: 1px solid #e6e9ee; border-left: 4px solid #0e7c7b;
    border-radius: 10px; padding: 20px 24px; margin-bottom: 16px;
}
.rec-title { font-size: 15px; font-weight: 700; color: #101b2d; margin-bottom: 8px; }
.rec-detail { font-size: 13.5px; color: #3d4759; line-height: 1.55; }

button[data-baseweb="tab"] { font-size: 13px; font-weight: 600; padding: 8px 16px; }
button[data-baseweb="tab"][aria-selected="true"] { color: #0e7c7b; border-bottom: 2px solid #0e7c7b; }
.stDataFrame { border-radius: 8px; overflow: hidden; }
hr { border-color: #e6e9ee; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)


def style_chart(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(t=24, b=16, l=8, r=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=12, color="#4a5568"),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#f0f2f5", zeroline=False, tickfont=dict(size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hoverlabel=dict(bgcolor="white", bordercolor="#e6e9ee", font=dict(family="Inter", size=12)),
    )
    return fig


SEGMENT_ORDER = ["Champions", "Loyal", "Needs Attention", "At Risk", "New", "Dormant"]
SEGMENT_COLORS = {
    "Champions": "#0e7c7b", "Loyal": "#3fa796", "Needs Attention": "#e8b23a",
    "At Risk": "#e0743a", "New": "#5b8def", "Dormant": "#94a1b5",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    merchants_path = os.path.join(BASE_DIR, "data", "merchants.csv")
    if not os.path.exists(merchants_path):
        subprocess.run(["python", "generate_data.py"], check=True, cwd=BASE_DIR)
    return load_all(os.path.join(BASE_DIR, "data"))


@st.cache_data
def run_analysis(_merchants, _transactions):
    rfm = build_rfm_segments(_merchants, _transactions)
    clv, auc = build_clv_table(rfm, _merchants, _transactions)
    summary = build_segment_summary(clv, _transactions)
    recs = generate_recommendations(summary)
    return clv, summary, recs, auc


merchants_df, transactions_df = load_data()
clv_df, summary_df, recommendations, churn_auc = run_analysis(merchants_df, transactions_df)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>💳 Payment Merchant Segmentation & Lifetime Value</h1>
    <p>RFM segmentation, CLV estimation, and payment failure exposure across a simulated merchant book</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ──────────────────────────────────────────────────────────────────
total_merchants = len(clv_df)
total_clv = clv_df["total_clv"].sum()
platform_failure_rate = transactions_df["status"].eq("Failed").mean()
retain_value = clv_df[clv_df["segment"].isin(["Champions", "Loyal", "At Risk"])]["total_clv"].sum()
at_risk_value = clv_df[clv_df["segment"] == "At Risk"]["total_clv"].sum()

c1, c2, c3, c4, c5 = st.columns(5)


def kpi_card(col, icon, label, value, sub):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{icon} {label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


kpi_card(c1, "🏪", "Total Merchants", f"{total_merchants:,}", "In simulated portfolio")
kpi_card(c2, "💰", "Total Platform CLV", f"${total_clv:,.0f}", f"At a {TAKE_RATE:.1%} take rate")
kpi_card(c3, "⚠️", "Platform Failure Rate", f"{platform_failure_rate:.1%}", "Across all transaction attempts")
kpi_card(c4, "🏆", "Value in Champions/Loyal/At Risk", f"${retain_value:,.0f}", "Segments worth actively retaining")
kpi_card(c5, "📉", "At Risk Segment Value", f"${at_risk_value:,.0f}", "High historic value, gone quiet")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Segment Overview",
    "💵 Lifetime Value",
    "⚠️ Failure Rate by Segment",
    "✅ Recommendations",
])

# ── Tab 1: Segment Overview ─────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Merchant Segments (RFM)</div>', unsafe_allow_html=True)
    st.caption(
        "Merchants are scored on Recency, Frequency, and Monetary value (successful transactions only), "
        "each split into quintiles, then assigned to a segment based on how recent and how valuable they are."
    )

    left, right = st.columns([1, 1.3])

    with left:
        seg_counts = clv_df["segment"].value_counts().reindex(SEGMENT_ORDER).reset_index()
        seg_counts.columns = ["segment", "merchant_count"]
        fig = px.bar(
            seg_counts, x="segment", y="merchant_count", color="segment",
            color_discrete_map=SEGMENT_COLORS, text="merchant_count",
        )
        fig.update_traces(textposition="outside", showlegend=False)
        fig.update_layout(xaxis_title=None, yaxis_title="Merchants")
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        sample = clv_df.sample(min(1200, len(clv_df)), random_state=1)
        fig2 = px.scatter(
            sample, x="recency_days", y="frequency", size="monetary", color="segment",
            color_discrete_map=SEGMENT_COLORS, opacity=0.65,
            labels={"recency_days": "Days Since Last Transaction", "frequency": "Successful Transactions"},
            hover_data={"merchant_id": True, "monetary": ":$,.0f"},
        )
        fig2.update_layout(legend_title=None)
        st.plotly_chart(style_chart(fig2), use_container_width=True)

    st.markdown('<div class="section-header">What Each Segment Means</div>', unsafe_allow_html=True)
    seg_def_df = pd.DataFrame({
        "Segment": list(SEGMENT_DEFINITIONS.keys()),
        "Definition": list(SEGMENT_DEFINITIONS.values()),
    }).set_index("Segment").reindex(SEGMENT_ORDER)
    st.dataframe(seg_def_df, use_container_width=True)

# ── Tab 2: Lifetime Value ───────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">CLV Distribution</div>', unsafe_allow_html=True)
    st.caption(
        f"CLV is platform revenue, not merchant GMV: total transaction value x {TAKE_RATE:.1%} assumed take rate. "
        f"Total CLV = historic platform revenue earned to date + predicted future revenue for merchants still active. "
        f"The churn model behind the prediction is a logistic regression scoring {churn_auc:.2f} AUC on held-out merchants."
    )

    left, right = st.columns([1.3, 1])

    with left:
        fig3 = px.histogram(
            clv_df, x="total_clv", nbins=60, color_discrete_sequence=["#0e7c7b"],
            labels={"total_clv": "Total CLV ($)"},
        )
        fig3.update_layout(yaxis_title="Merchants")
        st.plotly_chart(style_chart(fig3), use_container_width=True)
        st.caption("Distribution is long-tailed, as expected: a minority of merchants carry a large share of platform value.")

    with right:
        fig4 = px.box(
            clv_df, x="segment", y="total_clv", color="segment",
            color_discrete_map=SEGMENT_COLORS, category_orders={"segment": SEGMENT_ORDER},
            points=False, labels={"total_clv": "Total CLV ($)"},
        )
        fig4.update_layout(xaxis_title=None, showlegend=False)
        st.plotly_chart(style_chart(fig4), use_container_width=True)

    st.markdown('<div class="section-header">Top Merchants by Lifetime Value</div>', unsafe_allow_html=True)
    top_merchants = clv_df.sort_values("total_clv", ascending=False).head(15)[
        ["merchant_id", "industry", "segment", "historic_platform_revenue", "predicted_future_revenue", "total_clv"]
    ].rename(columns={
        "historic_platform_revenue": "Historic Revenue ($)",
        "predicted_future_revenue": "Predicted Future Revenue ($)",
        "total_clv": "Total CLV ($)",
    })
    st.dataframe(top_merchants.style.format({
        "Historic Revenue ($)": "${:,.0f}",
        "Predicted Future Revenue ($)": "${:,.0f}",
        "Total CLV ($)": "${:,.0f}",
    }), use_container_width=True, hide_index=True)

# ── Tab 3: Failure Rate by Segment ──────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Failure Rate by Segment</div>', unsafe_allow_html=True)
    st.caption(
        "Failure rate is calculated across all transaction attempts (successful and failed), which is the standard "
        "way decline rate is measured. The dashed line marks the platform-wide average for comparison."
    )

    ordered_summary = summary_df.set_index("segment").reindex(SEGMENT_ORDER).reset_index()

    left, right = st.columns(2)

    with left:
        fig5 = px.bar(
            ordered_summary, x="segment", y="failure_rate", color="segment",
            color_discrete_map=SEGMENT_COLORS, text=ordered_summary["failure_rate"].map(lambda v: f"{v:.1%}"),
        )
        fig5.add_hline(
            y=platform_failure_rate, line_dash="dash", line_color="#3d4759",
            annotation_text="Platform average", annotation_position="top left",
        )
        fig5.update_traces(textposition="outside", showlegend=False)
        fig5.update_layout(xaxis_title=None, yaxis_title="Failure Rate", yaxis_tickformat=".0%")
        st.plotly_chart(style_chart(fig5), use_container_width=True)

    with right:
        fig6 = px.bar(
            ordered_summary.sort_values("value_weighted_failure_exposure", ascending=True),
            x="value_weighted_failure_exposure", y="segment", orientation="h",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            labels={"value_weighted_failure_exposure": "CLV x Failure Rate ($)"},
        )
        fig6.update_traces(showlegend=False)
        fig6.update_layout(yaxis_title=None)
        st.plotly_chart(style_chart(fig6), use_container_width=True)
        st.caption("Value-weighted failure exposure = total segment CLV x segment failure rate. It surfaces where the most lifetime value sits behind unreliable payments, not just which segment fails the most.")

    st.markdown('<div class="section-header">Full Segment Summary</div>', unsafe_allow_html=True)
    display_summary = ordered_summary[[
        "segment", "merchant_count", "total_clv", "avg_clv", "failure_rate",
        "platform_avg_failure_rate", "value_weighted_failure_exposure",
    ]].rename(columns={
        "segment": "Segment", "merchant_count": "Merchants", "total_clv": "Total CLV ($)",
        "avg_clv": "Avg CLV ($)", "failure_rate": "Failure Rate", "platform_avg_failure_rate": "Platform Avg",
        "value_weighted_failure_exposure": "Value-Weighted Exposure ($)",
    })
    st.dataframe(display_summary.style.format({
        "Total CLV ($)": "${:,.0f}", "Avg CLV ($)": "${:,.0f}", "Failure Rate": "{:.1%}",
        "Platform Avg": "{:.1%}", "Value-Weighted Exposure ($)": "${:,.0f}",
    }), use_container_width=True, hide_index=True)

# ── Tab 4: Recommendations ──────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">What I Would Recommend</div>', unsafe_allow_html=True)
    st.caption(
        "These are generated directly from the segment summary above: recommendation 1 ranks valuable segments "
        "by value-weighted failure exposure and keeps only ones failing above the platform average; "
        "recommendation 2 looks specifically at the At Risk segment, since it already shows the recency drop-off "
        "pattern of churn in progress."
    )

    for i, rec in enumerate(recommendations, start=1):
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-title">{i}. {rec['title']}</div>
            <div class="rec-detail">{rec['detail']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Method Notes</div>', unsafe_allow_html=True)
    with st.expander("How segments, CLV, and exposure are calculated"):
        st.markdown(f"""
- **RFM segmentation**: Recency (days since last successful transaction), Frequency (count of successful
  transactions), Monetary (total successful transaction value), each scored into quintiles, combined into
  six named segments.
- **CLV**: platform revenue, not merchant GMV. `historic revenue = successful transaction volume x {TAKE_RATE:.1%}`.
  Predicted future revenue only applies to merchants without a 60+ day transaction gap, using a logistic
  regression churn model (AUC {churn_auc:.2f} on held-out merchants) and the standard `expected lifetime =
  1 / monthly churn probability` relationship, capped at 60 months.
- **Failure exposure**: `segment total CLV x segment failure rate`, so a segment's priority reflects both how
  often it fails and how much value is actually behind that failure rate.
        """)
