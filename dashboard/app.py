"""
Phase 8 - Streamlit dashboard.

Reads only the CSVs written by Phases 3-7. Nothing is retrained or refitted
here, so the app starts in about a second and cannot disagree with the
analysis that produced it.

That separation is deliberate. A dashboard that retrains on load is slow,
non-deterministic, and will eventually show a number that contradicts the
notebook it came from.

Run from the project root:
    streamlit run dashboard/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
MODELS = PROJECT_ROOT / "models"

st.set_page_config(page_title="Dynamic Pricing Intelligence",
                   page_icon="H", layout="wide")

ACCENT = "#2E5E4E"
WARN = "#B4472A"


# ==========================================================================
# Data loading
# ==========================================================================
@st.cache_data
def load(name: str, folder: Path = REPORTS, dates=None) -> pd.DataFrame:
    path = folder / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=dates or [])


def require(frames: dict) -> bool:
    """Fail loudly and usefully if a phase has not been run yet."""
    missing = [name for name, frame in frames.items() if frame.empty]
    if missing:
        st.error(f"Missing data: {', '.join(missing)}. "
                 "Run the corresponding phase script first.")
        return False
    return True


recommendations = load("price_recommendations", dates=["stay_date"])
explanations = load("recommendation_explanations", dates=["stay_date"])
scenarios = load("price_scenarios", dates=["stay_date"])
hotel_kpis = load("hotel_kpis")
model_comparison = load("model_comparison")
elasticity_validation = load("elasticity_validation")
robustness = load("elasticity_robustness")
shap_importance = load("shap_feature_importance")
strategies = load("strategy_comparison")
eda_findings = load("eda_findings")

HOTEL_NAMES = {
    "HTL001": "Harborview Grand", "HTL002": "Lakeside Inn",
    "HTL003": "Beachcomber Resort", "HTL004": "Metro Central",
    "HTL005": "Canyon Lodge",
}
ROOM_NAMES = {"RT_STD": "Standard", "RT_DLX": "Deluxe", "RT_STE": "Suite"}


def prettify(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "hotel_id" in out:
        out["hotel"] = out["hotel_id"].map(HOTEL_NAMES).fillna(out["hotel_id"])
    if "room_type_id" in out:
        out["room_type"] = out["room_type_id"].map(ROOM_NAMES).fillna(
            out["room_type_id"])
    return out


# ==========================================================================
# Sidebar
# ==========================================================================
st.sidebar.title("Dynamic Pricing")
st.sidebar.caption("Hotel revenue optimization")

page = st.sidebar.radio("Page", [
    "Executive overview",
    "Recommendations",
    "Rate simulator",
    "Model performance",
])

st.sidebar.divider()
st.sidebar.caption(
    "All figures are **simulated** under the project's demand model. "
    "Uplift is a model estimate, not a measured result."
)


# ==========================================================================
# Page 1 - Executive overview
# ==========================================================================
def page_overview():
    st.title("Executive overview")
    if not require({"recommendations": recommendations}):
        return

    rec = prettify(recommendations)
    baseline_rev = rec["baseline_revenue"].sum()
    optimized_rev = rec["expected_revenue"].sum()
    uplift = optimized_rev - baseline_rev

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open nights priced", f"{len(rec):,}",
              f"{rec['stay_date'].nunique()} stay dates")
    c2.metric("Revenue at current rates", f"${baseline_rev/1e6:.2f}M")
    c3.metric("Revenue if optimized", f"${optimized_rev/1e6:.2f}M",
              f"{uplift/baseline_rev*100:+.1f}%")
    c4.metric("Opportunity", f"${uplift/1e3:,.0f}K",
              "simulated", delta_color="off")

    st.caption(
        "The baseline and the optimized figure both come from the same demand "
        "model. A live A/B test on held-out stay dates is the only way to "
        "measure the real uplift."
    )
    st.divider()

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Opportunity by stay date")
        by_date = (rec.groupby("stay_date")
                   .agg(baseline=("baseline_revenue", "sum"),
                        optimized=("expected_revenue", "sum"),
                        occupancy=("baseline_occupancy", "mean"))
                   .reset_index())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=by_date["stay_date"], y=by_date["baseline"],
                                 name="Current rates", line=dict(color="#999")))
        fig.add_trace(go.Scatter(x=by_date["stay_date"], y=by_date["optimized"],
                                 name="Optimized", line=dict(color=ACCENT),
                                 fill="tonexty"))
        fig.update_layout(height=340, margin=dict(t=10, b=10),
                          yaxis_title="Expected revenue ($)",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Where the opportunity sits")
        bands = pd.cut(rec["baseline_occupancy"], [-0.01, 0.5, 0.7, 0.85, 1.01],
                       labels=["<50%", "50-70%", "70-85%", ">85%"])
        by_band = (rec.groupby(bands, observed=True)
                   .agg(uplift=("revenue_uplift", "sum"),
                        nights=("rate", "size"),
                        rate_change=("rate_change_pct", "mean"))
                   .reset_index())
        by_band.columns = ["Occupancy on books", "Uplift ($)", "Nights",
                           "Mean rate change (%)"]
        fig = px.bar(by_band, x="Occupancy on books", y="Uplift ($)",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Portfolio performance (historical)")
        if not hotel_kpis.empty:
            kpi = hotel_kpis.copy()
            kpi["ADR"] = kpi["adr"].round(0)
            kpi["Occupancy %"] = (kpi["occupancy"] * 100).round(1)
            kpi["RevPAR"] = kpi["revpar"].round(0)
            st.dataframe(kpi[["hotel_name", "ADR", "Occupancy %", "RevPAR"]]
                         .sort_values("RevPAR", ascending=False),
                         hide_index=True, width="stretch")
            st.caption(
                "Occupancy is near-identical across hotels because the legacy "
                "pricing rule converts demand into **rate**, not occupancy. "
                "ADR varies by 35% across days of the week; occupancy barely moves."
            )

    with c2:
        st.subheader("Strategy comparison")
        if not strategies.empty:
            frame = strategies.copy()
            frame["Revenue ($M)"] = (frame["revenue"] / 1e6).round(2)
            frame["Profit ($M)"] = (frame["profit"] / 1e6).round(2)
            frame["Mean rate change"] = frame["mean_rate_change_pct"].map(
                lambda v: f"{v:+.1f}%")
            st.dataframe(frame[["strategy", "Mean rate change",
                                "Revenue ($M)", "Profit ($M)"]],
                         hide_index=True, width="stretch")
            st.caption(
                "Revenue maximization over-discounts: profit maximization halves "
                "the average rate cut and still delivers marginally more profit."
            )


# ==========================================================================
# Page 2 - Recommendations
# ==========================================================================
def page_recommendations():
    st.title("Rate recommendations")
    if not require({"recommendations": recommendations,
                    "explanations": explanations}):
        return

    rec = prettify(recommendations)
    exp = prettify(explanations)

    f1, f2, f3 = st.columns(3)
    hotels = f1.multiselect("Hotel", sorted(rec["hotel"].unique()),
                            default=sorted(rec["hotel"].unique()))
    rooms = f2.multiselect("Room type", ["Standard", "Deluxe", "Suite"],
                           default=["Standard", "Deluxe", "Suite"])
    action = f3.selectbox("Action", ["All", "Increase", "Decrease", "Hold"])

    view = rec[rec["hotel"].isin(hotels) & rec["room_type"].isin(rooms)]
    if action == "Increase":
        view = view[view["rate_change_pct"] > 0.5]
    elif action == "Decrease":
        view = view[view["rate_change_pct"] < -0.5]
    elif action == "Hold":
        view = view[view["rate_change_pct"].abs() <= 0.5]

    if view.empty:
        st.warning("No nights match these filters.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Nights shown", f"{len(view):,}")
    c2.metric("Mean rate change", f"{view['rate_change_pct'].mean():+.1f}%")
    c3.metric("Revenue opportunity", f"${view['revenue_uplift'].sum():,.0f}")

    st.divider()

    table = view.copy()
    # Rounded to the dollar on purpose: Phase 7 found the revenue optimum is
    # flat, so cents imply a precision the model does not have.
    table["Current"] = table["current_rate"].round(0).map("${:,.0f}".format)
    table["Recommended"] = table["rate"].round(0).map("${:,.0f}".format)
    table["Change"] = table["rate_change_pct"].map("{:+.1f}%".format)
    table["Occupancy"] = table["baseline_occupancy"].map("{:.0%}".format)
    table["Uplift"] = table["revenue_uplift"].map("${:,.0f}".format)
    table["Date"] = table["stay_date"].dt.strftime("%a %d %b")

    st.dataframe(
        table[["Date", "hotel", "room_type", "Current", "Recommended",
               "Change", "Occupancy", "Uplift"]]
        .sort_values("Uplift", key=lambda s: s.str.replace(r"[$,]", "",
                                                           regex=True).astype(float),
                     ascending=False),
        hide_index=True, width="stretch", height=320)

    st.divider()
    st.subheader("Why this recommendation?")

    options = view.sort_values("revenue_uplift", ascending=False)
    labels = [f"{r.stay_date.strftime('%d %b')} - {r.hotel} - {r.room_type}"
              for r in options.itertuples()]
    choice = st.selectbox("Select a night", labels)
    row = options.iloc[labels.index(choice)]

    detail = exp[(exp["hotel_id"] == row["hotel_id"])
                 & (exp["room_type_id"] == row["room_type_id"])
                 & (exp["stay_date"] == row["stay_date"])]

    if detail.empty:
        st.info("No stored explanation for this night.")
        return
    detail = detail.iloc[0]

    left, right = st.columns([2, 3])

    with left:
        st.metric("Recommended rate", f"${detail['recommended_rate']:,.0f}",
                  f"{detail['rate_change_pct']:+.1f}% vs ${detail['current_rate']:,.0f}")
        st.write(f"""
**Why:**

- Measured elasticity **{detail['elasticity']:.2f}** — a 10% rate change moves
  bookings by about **{abs(detail['elasticity']) * 10:.0f}%**
- Night is **{detail['occupancy_on_books']:.0%} sold**, with
  **{detail['headroom_pct']:.0%}** of inventory still to fill
- Constraint: **{detail['constraint_binding']}**
- Expected: **{detail['expected_pickup']:.0f}** more rooms,
  **${detail['revenue_uplift']:,.0f}** additional revenue
- Cost of doing nothing: **${detail['cost_of_inaction']:,.0f}**
        """)
        if detail["rates_within_1pct_of_best"] >= 3:
            st.info(
                f"{int(detail['rates_within_1pct_of_best'])} of 19 candidate rates "
                "land within 1% of best revenue. The optimum is flat — treat this "
                "as a range, not a precise figure."
            )

    with right:
        night = scenarios[(scenarios["hotel_id"] == row["hotel_id"])
                          & (scenarios["room_type_id"] == row["room_type_id"])
                          & (scenarios["stay_date"] == row["stay_date"])]
        if not night.empty:
            night = night.sort_values("rate")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=night["rate"], y=night["expected_revenue"],
                                     mode="lines+markers", name="Revenue",
                                     line=dict(color=ACCENT)))
            fig.add_vline(x=detail["current_rate"], line_dash="dash",
                          line_color="#999", annotation_text="current")
            fig.add_vline(x=detail["recommended_rate"], line_color=ACCENT,
                          annotation_text="recommended")
            fig.update_layout(height=360, margin=dict(t=30, b=10),
                              xaxis_title="Rate ($)",
                              yaxis_title="Expected revenue ($)")
            st.plotly_chart(fig, width="stretch")


# ==========================================================================
# Page 3 - Rate simulator
# ==========================================================================
def page_simulator():
    st.title("Rate simulator")
    st.caption(
        "Pick a night and move the rate. Curves are precomputed by the Phase 6 "
        "forward simulation — each point rolls the night day by day to arrival, "
        "respecting remaining capacity."
    )
    if not require({"scenarios": scenarios}):
        return

    sc = prettify(scenarios)
    rec = prettify(recommendations)

    c1, c2, c3 = st.columns(3)
    hotel = c1.selectbox("Hotel", sorted(sc["hotel"].unique()))
    room = c2.selectbox("Room type", ["Standard", "Deluxe", "Suite"])
    dates = sorted(sc[(sc["hotel"] == hotel) & (sc["room_type"] == room)]
                   ["stay_date"].unique())
    stay_date = c3.selectbox("Stay date", dates,
                             format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b %Y"))

    night = sc[(sc["hotel"] == hotel) & (sc["room_type"] == room)
               & (sc["stay_date"] == stay_date)].sort_values("rate")
    current = rec[(rec["hotel"] == hotel) & (rec["room_type"] == room)
                  & (rec["stay_date"] == stay_date)]
    if night.empty or current.empty:
        st.warning("No scenario data for this night.")
        return
    current = current.iloc[0]

    # Slider options are plain rounded floats. Deliberately no format_func:
    # Streamlit stores the option value in session state, and a formatted
    # string ("$396") cannot round-trip back to the option it came from.
    # The dollar sign lives in the label instead.
    rates = night["rate"].round(0).tolist()
    default = min(rates, key=lambda r: abs(r - current["rate"]))

    chosen = st.select_slider("Rate ($)", options=rates, value=default)
    picked = night.iloc[(night["rate"] - chosen).abs().argmin()]
    base = night.iloc[(night["rate_multiplier"] - 1.0).abs().argmin()]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rate", f"${picked['rate']:,.0f}",
              f"{(picked['rate']/base['rate']-1)*100:+.1f}%")
    m2.metric("Expected pickup", f"{picked['expected_pickup']:,.0f} rooms",
              f"{picked['expected_pickup']-base['expected_pickup']:+,.0f}")
    m3.metric("Final occupancy", f"{picked['final_occupancy']:.0%}",
              f"{(picked['final_occupancy']-base['final_occupancy'])*100:+.1f} pts")
    m4.metric("Expected revenue", f"${picked['expected_revenue']:,.0f}",
              f"${picked['expected_revenue']-base['expected_revenue']:+,.0f}")

    st.divider()
    charts = st.columns(3)
    for col, (field, title) in zip(charts, [
            ("expected_revenue", "Revenue ($)"),
            ("expected_pickup", "Rooms booked"),
            ("final_occupancy", "Final occupancy")]):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=night["rate"], y=night[field],
                                 mode="lines+markers", line=dict(color=ACCENT)))
        fig.add_vline(x=picked["rate"], line_color=WARN)
        fig.update_layout(height=280, margin=dict(t=30, b=10), title=title,
                          xaxis_title="Rate ($)")
        col.plotly_chart(fig, width="stretch")

    st.caption(
        f"Elasticity for this segment: **{picked['elasticity']:.2f}**. "
        "Note the revenue curve is flat near its peak — several rates perform "
        "within 1% of the best."
    )


# ==========================================================================
# Page 4 - Model performance
# ==========================================================================
def page_models():
    st.title("Model performance")

    tab1, tab2, tab3 = st.tabs(["Demand forecast", "Causal elasticity",
                                "Feature drivers"])

    with tab1:
        st.subheader("Model comparison (test split)")
        if model_comparison.empty:
            st.info("Run `python src/demand_model.py` first.")
        else:
            frame = model_comparison.sort_values("poisson_deviance")
            st.dataframe(frame, hide_index=True, width="stretch")
            c1, c2 = st.columns(2)
            fig = px.bar(frame, x="model", y="poisson_deviance",
                         color_discrete_sequence=[ACCENT],
                         title="Poisson deviance (lower is better)")
            fig.update_layout(height=320)
            c1.plotly_chart(fig, width="stretch")
            fig = px.bar(frame, x="model", y="fit_seconds",
                         color_discrete_sequence=["#888"],
                         title="Fit time (seconds)")
            fig.update_layout(height=320)
            c2.plotly_chart(fig, width="stretch")
            st.caption(
                "The three non-linear models are within 0.005 deviance of each "
                "other — statistically tied. XGBoost was selected on fit time, "
                "which is a legitimate tiebreak when accuracy is indistinguishable."
            )

    with tab2:
        st.subheader("Recovering the true elasticity")
        if elasticity_validation.empty:
            st.info("Run `python src/elasticity.py` first.")
        else:
            val = elasticity_validation
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean absolute error",
                      f"{val['error'].abs().mean():.3f}")
            c2.metric("95% CI coverage", f"{val['covered'].mean():.0%}")
            c3.metric("Correct sign", f"{(val['elasticity'] < 0).mean():.0%}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=val["true_elasticity"], y=val["elasticity"],
                mode="markers", marker=dict(size=9, color=ACCENT),
                error_y=dict(type="data",
                             array=val["ci_high"] - val["elasticity"],
                             arrayminus=val["elasticity"] - val["ci_low"]),
                name="estimated"))
            lims = [-2.3, -0.5]
            fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                     line=dict(dash="dash", color="#999"),
                                     name="perfect recovery"))
            fig.update_layout(height=400, xaxis_title="True elasticity",
                              yaxis_title="Estimated elasticity")
            st.plotly_chart(fig, width="stretch")

            if not robustness.empty:
                st.subheader("Robustness")
                st.dataframe(robustness, hide_index=True,
                             width="stretch")
                st.caption(
                    "The placebo row applies the identical estimator to "
                    "non-randomized rates labelled as if they were an experiment. "
                    "It returns the **wrong sign** — which is what proves the real "
                    "result comes from the experimental design, not the estimator."
                )

    with tab3:
        st.subheader("What drives the forecast")
        if shap_importance.empty:
            st.info("Run `python src/explain.py` first.")
        else:
            top = shap_importance.head(12)
            fig = px.bar(top, y="label", x="mean_abs_shap", color="category",
                         orientation="h")
            fig.update_layout(height=420, yaxis=dict(autorange="reversed"),
                              xaxis_title="Mean |SHAP| (log space)",
                              yaxis_title="")
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "The model predicts in log space, so SHAP values are "
                "**multiplicative**: exp(0.7) means demand roughly doubles. "
                "Price contributes only ~4% — not because price is unimportant "
                "(measured elasticity is -1.5) but because the legacy rule moved "
                "price with demand, making it redundant for *forecasting*. That "
                "redundancy is exactly why the optimizer uses the causal "
                "elasticity instead."
            )


# ==========================================================================
PAGES = {
    "Executive overview": page_overview,
    "Recommendations": page_recommendations,
    "Rate simulator": page_simulator,
    "Model performance": page_models,
}
PAGES[page]()
