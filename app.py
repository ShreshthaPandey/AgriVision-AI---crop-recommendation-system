import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Crop Recommendation", layout="wide")


@st.cache_resource
def load():
    model = joblib.load("final_model.pkl")
    d = pd.read_csv("final_data.csv")
    ref = pd.read_csv("crop_reference.csv").set_index("Crop")
    sw = pd.read_csv("state_weather.csv")
    return model, d, ref, sw


model, d, ref, sw = load()

cats = {c: sorted(d[c].unique()) for c in ["State", "District", "Crop", "Season"]}
q = d.groupby("Crop")["Yield"].quantile([0.30, 0.60]).unstack()
q.columns = ["q_low", "q_high"]
next_year = int(d["Crop_Year"].max()) + 1

# Rough (price Rs/quintal, cost Rs/ha, effort) for crops missing in crop_reference.csv
extra_ref = {
    "Arecanut": (45000, 150000, "High"), "Cashewnut": (12000, 60000, "Medium"),
    "Black Pepper": (50000, 150000, "High"), "Cardamom": (120000, 250000, "High"),
    "Cowpea(Lobia)": (5500, 22000, "Low"), "Guar Seed": (5500, 20000, "Low"),
    "Horse-Gram": (5000, 18000, "Low"), "Khesari": (4000, 18000, "Low"),
    "Moth": (6000, 18000, "Low"), "Tobacco": (15000, 80000, "High"),
}

# crop: (min temp, max temp, min rain mm per season) - approximate
ideal = {
    "Rice": (20, 35, 600), "Wheat": (10, 25, 100), "Maize": (18, 32, 300),
    "Gram": (15, 30, 60), "Rapeseed &Mustard": (10, 25, 60),
    "Groundnut": (22, 32, 300), "Soyabean": (20, 32, 400),
    "Sugarcane": (20, 35, 800), "Potato": (10, 25, 100), "Onion": (13, 30, 100),
    "Bajra": (25, 35, 250), "Jowar": (25, 35, 300), "Ragi": (20, 32, 300),
    "Barley": (10, 25, 80), "Urad": (22, 35, 300), "Moong(Green Gram)": (22, 35, 300),
    "Masoor": (10, 25, 60), "Arhar/Tur": (20, 35, 400), "Sunflower": (20, 32, 250),
    "Sesamum": (22, 35, 250), "Safflower": (15, 30, 100), "Niger Seed": (18, 30, 300),
    "Castor Seed": (20, 35, 300), "Linseed": (10, 25, 80), "Garlic": (10, 25, 100),
    "Ginger": (20, 32, 600), "Turmeric": (20, 32, 600), "Dry Chillies": (20, 32, 400),
    "Coriander": (15, 28, 100), "Sweet Potato": (20, 32, 400), "Tapioca": (22, 35, 600),
    "Banana": (20, 35, 800), "Arecanut": (20, 32, 1000), "Cashewnut": (22, 35, 600),
    "Black Pepper": (20, 32, 1000), "Cardamom": (15, 28, 1000), "Cowpea(Lobia)": (22, 35, 300),
    "Guar Seed": (25, 35, 250), "Horse-Gram": (22, 35, 250), "Khesari": (12, 28, 100),
    "Moth": (25, 38, 150), "Tobacco": (18, 32, 300),
}


def weather_fit(crop, temp, rain):
    if temp is None or crop not in ideal:
        return "Not available"
    tmin, tmax, rmin = ideal[crop]
    ok = int(tmin <= temp <= tmax) + int(rain >= rmin)
    return ["Poor", "Moderate", "Good"][ok]


def color_status(v):
    if v == "Profit":
        return "color: green; font-weight: bold"
    if v == "Loss":
        return "color: red; font-weight: bold"
    return ""


def color_amount(v):
    if pd.isna(v):
        return ""
    return "color: green" if v > 0 else "color: red"


st.title("Crop Recommendation System")

c1, c2, c3 = st.columns(3)
state = c1.selectbox("State", sorted(d["State"].unique()))
district = c2.selectbox(
    "District", sorted(d[d["State"] == state]["District"].unique()))
seasons = sorted(
    d[(d["State"] == state) & (d["District"] == district)]["Season"].unique())
season = c3.selectbox("Season", seasons)

w = sw[(sw["State"] == state) & (sw["Season"] == season)]
temp = rain = None
if not w.empty:
    temp, rain = float(w["Temp"].iloc[0]), float(w["Rain"].iloc[0])
    st.info(f"{state} - {season}: Rain {rain:.0f} mm | Temp {temp:.1f} C | "
            f"Humidity {w['Humidity'].iloc[0]:.0f}%")

if st.button("Recommend"):
    sub = d[(d["State"] == state) & (d["District"] == district) & (d["Season"] == season)]
    rows = []
    for crop, g in sub.groupby("Crop"):
        g = g.sort_values("Crop_Year")
        X = pd.DataFrame([{
            "State": state, "District": district, "Crop": crop, "Season": season,
            "Crop_Year": next_year,
            "Prev_Yield": g["Yield"].iloc[-1],
            "Prev3_Avg": g["Yield"].tail(3).mean()}])
        for c in ["State", "District", "Crop", "Season"]:
            X[c] = pd.Categorical(X[c], categories=cats[c])
        y = float(model.predict(X)[0])

        lo, hi = q.loc[crop, "q_low"], q.loc[crop, "q_high"]
        perf = "Low" if y <= lo else ("Medium" if y <= hi else "High")

        price = cost = None
        effort = "N/A"
        if crop in ref.index and pd.notna(ref.loc[crop, "Price"]):
            price, cost = ref.loc[crop, "Price"], ref.loc[crop, "Cost"]
            effort = ref.loc[crop, "Effort"]
        elif crop in extra_ref:
            price, cost, effort = extra_ref[crop]

        profit = y * 10 * price - cost if price is not None else None
        profit_val = round(profit) if profit is not None else None
        if profit_val is None:
            status = "N/A"
        elif profit_val > 0:
            status = "Profit"
        else:
            status = "Loss"

        rows.append({
            "Crop": crop,
            "Predicted Yield (t/ha)": round(y, 2),
            "Performance": perf,
            "Effort": effort,
            "Weather Fit": weather_fit(crop, temp, rain),
            "Profit / Loss": status,
            "Estimated Profit/Loss (Rs/ha)": profit_val,
        })

    if not rows:
        st.warning("Is district + season ke liye data nahi hai.")
    else:
        out = pd.DataFrame(rows).sort_values(
            "Estimated Profit/Loss (Rs/ha)", ascending=False, na_position="last")

        styled = (out.style
                  .map(color_status, subset=["Profit / Loss"])
                  .map(color_amount, subset=["Estimated Profit/Loss (Rs/ha)"])
                  .format({"Estimated Profit/Loss (Rs/ha)": "{:,.0f}"}, na_rep="N/A"))

        st.dataframe(styled, use_container_width=True, hide_index=True)

        n_profit = int((out["Profit / Loss"] == "Profit").sum())
        n_loss = int((out["Profit / Loss"] == "Loss").sum())
        st.caption(
            f"{n_profit} crops mein estimated profit, {n_loss} mein estimated loss. "
            "Price aur cost approximate hain, actual farm par badal sakte hain. "
            "High-profit crops (Banana, Ginger, Turmeric) mein investment aur risk "
            "bhi zyada hota hai.")