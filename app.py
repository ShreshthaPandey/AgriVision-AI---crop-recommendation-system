import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="KisanMitra", page_icon="🌾", layout="wide")


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
q.columns = ["lo", "hi"]
next_year = int(d["Crop_Year"].max()) + 1

extra_ref = {
    "Arecanut": (45000, 150000, "High"), "Cashewnut": (12000, 60000, "Medium"),
    "Black Pepper": (50000, 150000, "High"), "Cardamom": (120000, 250000, "High"),
    "Cowpea(Lobia)": (5500, 22000, "Low"), "Guar Seed": (5500, 20000, "Low"),
    "Horse-Gram": (5000, 18000, "Low"), "Khesari": (4000, 18000, "Low"),
    "Moth": (6000, 18000, "Low"), "Tobacco": (15000, 80000, "High"),
}
low_conf = {"Cardamom", "Black Pepper", "Cashewnut", "Niger Seed", "Castor Seed",
            "Sesamum", "Tobacco", "Safflower", "Dry Chillies"}
# crop: (min temp, max temp, min rain mm/season) - approximate, verify (ICAR / FAO EcoCrop)
ideal = {
    "Rice": (20, 35, 600), "Wheat": (10, 25, 100), "Maize": (18, 32, 300), "Gram": (15, 30, 60),
    "Rapeseed &Mustard": (10, 25, 60), "Groundnut": (22, 32, 300), "Soyabean": (20, 32, 400),
    "Sugarcane": (20, 35, 800), "Potato": (10, 25, 100), "Onion": (13, 30, 100),
    "Bajra": (25, 35, 250), "Jowar": (25, 35, 300), "Ragi": (20, 32, 300), "Barley": (10, 25, 80),
    "Urad": (22, 35, 300), "Moong(Green Gram)": (22, 35, 300), "Masoor": (10, 25, 60),
    "Arhar/Tur": (20, 35, 400), "Sunflower": (20, 32, 250), "Sesamum": (22, 35, 250),
    "Safflower": (15, 30, 100), "Niger Seed": (18, 30, 300), "Castor Seed": (20, 35, 300),
    "Linseed": (10, 25, 80), "Garlic": (10, 25, 100), "Ginger": (20, 32, 600),
    "Turmeric": (20, 32, 600), "Dry Chillies": (20, 32, 400), "Coriander": (15, 28, 100),
    "Sweet Potato": (20, 32, 400), "Tapioca": (22, 35, 600), "Banana": (20, 35, 800),
    "Arecanut": (20, 32, 1000), "Cashewnut": (22, 35, 600), "Black Pepper": (20, 32, 1000),
    "Cardamom": (15, 28, 1000), "Cowpea(Lobia)": (22, 35, 300), "Guar Seed": (25, 35, 250),
    "Horse-Gram": (22, 35, 250), "Khesari": (12, 28, 100), "Moth": (25, 38, 150),
    "Tobacco": (18, 32, 300),
}

for k, v in {"page": 1, "rows": None, "sel": None, "ctx": None}.items():
    st.session_state.setdefault(k, v)


def go(n):
    st.session_state.page = n


def recommend(state, district, season):
    sub = d[(d["State"] == state) & (d["District"] == district) & (d["Season"] == season)]
    out = []
    for crop, g in sub.groupby("Crop"):
        g = g.sort_values("Crop_Year")
        X = pd.DataFrame([{"State": state, "District": district, "Crop": crop, "Season": season,
                           "Crop_Year": next_year, "Prev_Yield": g["Yield"].iloc[-1],
                           "Prev3_Avg": g["Yield"].tail(3).mean()}])
        for c in ["State", "District", "Crop", "Season"]:
            X[c] = pd.Categorical(X[c], categories=cats[c])
        y = float(model.predict(X)[0])
        perf = "Low" if y <= q.loc[crop, "lo"] else ("Medium" if y <= q.loc[crop, "hi"] else "High")
        price = cost = None
        effort = "N/A"
        if crop in ref.index and pd.notna(ref.loc[crop, "Price"]):
            price, cost, effort = float(ref.loc[crop, "Price"]), float(ref.loc[crop, "Cost"]), ref.loc[crop, "Effort"]
        elif crop in extra_ref:
            price, cost, effort = extra_ref[crop]
        profit = round(y * 10 * price - cost) if price is not None else None
        out.append({"Crop": crop, "Performance": perf, "Yield (t/ha)": round(y, 2), "Effort": effort,
                    "Profit / Loss": "N/A" if profit is None else ("Profit" if profit > 0 else "Loss"),
                    "Estimated (Rs/ha)": profit, "Confidence": "Low" if crop in low_conf else "OK",
                    "_price": price, "_cost": cost})
    out.sort(key=lambda r: (r["Estimated (Rs/ha)"] is None, -(r["Estimated (Rs/ha)"] or 0)))
    return out


def color(v):
    m = {"High": "green", "Profit": "green", "Medium": "orange", "Low": "red", "Loss": "red"}
    return f"color: {m[v]}; font-weight: bold" if v in m else ""


# ---------------- PAGE 1 ----------------
if st.session_state.page == 1:
    st.markdown("<h1 style='text-align:center;color:#2e7d32'>🌾 KisanMitra</h1>"
                "<h3 style='text-align:center'>Kya ugayein? Ab data batayega.</h3>", unsafe_allow_html=True)
    st.write("Bahut se kisan har season mein andaze se fasal chunte hain, aur kabhi kam yield, kabhi loss ho jata hai. "
             "KisanMitra aapke district ka purana yield data, mandi bhav aur mausam dekh kar batata hai ki "
             "kaunsi fasal kitna accha perform karegi, kitni mehnat lagegi aur kitna profit ho sakta hai.")
    a, b, c = st.columns(3)
    a.info("**Problem**\n\nGalat fasal ka chunav, mausam ka andaza nahi, kamai ka hisaab pehle se nahi pata.")
    b.success("**Solution**\n\nSirf State, District, Season daaliye. LightGBM model 12 saal ke district yield data se predict karta hai.")
    c.warning("**Kisan ko fayda**\n\nLow/Medium/High performance, effort, weather fit aur estimated profit/loss ek jagah.")
    st.button("Get Started", type="primary", on_click=go, args=(2,))

# ---------------- PAGE 2 ----------------
elif st.session_state.page == 2:
    st.title("Crop Recommendation")
    st.button("Back", on_click=go, args=(1,))
    c1, c2, c3 = st.columns(3)
    state = c1.selectbox("State", sorted(d["State"].unique()))
    district = c2.selectbox("District", sorted(d[d["State"] == state]["District"].unique()))
    season = c3.selectbox("Season", sorted(d[(d["State"] == state) & (d["District"] == district)]["Season"].unique()))
    if st.button("Recommend", type="primary"):
        st.session_state.rows = recommend(state, district, season)
        st.session_state.ctx = (state, district, season)
        st.session_state.sel = None

    rows = st.session_state.rows
    if rows is not None:
        if not rows:
            st.warning("Is district aur season ke liye data nahi hai.")
        else:
            df = pd.DataFrame(rows).drop(columns=["_price", "_cost"])
            sty = (df.style.map(color, subset=["Performance", "Profit / Loss"])
                   .format({"Estimated (Rs/ha)": "{:,.0f}"}, na_rep="N/A"))
            st.dataframe(sty, use_container_width=True, hide_index=True)
            st.caption("Estimated values. Price aur cost approximate hain.")
            pick = st.selectbox("Mausam aur kamai ka detail dekhne ke liye fasal chuno", [r["Crop"] for r in rows])
            if st.button("View details"):
                st.session_state.sel = pick
                go(3)
                st.rerun()

# ---------------- PAGE 3 ----------------
else:
    state, district, season = st.session_state.ctx
    r = next(x for x in st.session_state.rows if x["Crop"] == st.session_state.sel)
    st.title(f"{r['Crop']} - {district}, {season}")
    st.button("Back to list", on_click=go, args=(2,))
    left, right = st.columns(2)

    with left:
        st.subheader("Mausam (Open-Meteo data)")
        w = sw[(sw["State"] == state) & (sw["Season"] == season)]
        if w.empty:
            st.write("Is state/season ka weather data nahi mila.")
        else:
            t, rain, h = float(w["Temp"].iloc[0]), float(w["Rain"].iloc[0]), float(w["Humidity"].iloc[0])
            m1, m2, m3 = st.columns(3)
            m1.metric("Temp", f"{t:.1f} C"); m2.metric("Barish", f"{rain:.0f} mm"); m3.metric("Humidity", f"{h:.0f}%")
            if r["Crop"] in ideal:
                tmin, tmax, rmin = ideal[r["Crop"]]
                fit = ["Poor", "Moderate", "Good"][int(tmin <= t <= tmax) + int(rain >= rmin)]
                st.write(f"Ideal temperature: **{tmin}-{tmax} C** | Kam se kam barish: **{rmin} mm**")
                st.write(f"Weather fit: **{fit}**")
                st.progress(min(1.0, rain / max(rmin, 1)), text="Barish vs zarurat")
            else:
                st.caption("Is fasal ki ideal range available nahi.")

    with right:
        st.subheader("Kamai ka hisaab (per hectare)")
        if r["_price"] is None:
            st.write("Price/cost data nahi hai.")
        else:
            rev = round(r["Yield (t/ha)"] * 10 * r["_price"])
            st.write(f"Predicted yield: **{r['Yield (t/ha)']} t/ha** ({r['Yield (t/ha)'] * 10:.1f} quintal)")
            st.write(f"Price: **Rs {r['_price']:,.0f}/quintal** | Revenue: **Rs {rev:,}**")
            st.write(f"Kharcha: **Rs {r['_cost']:,.0f}**")
            st.metric(f"Estimated {r['Profit / Loss']}", f"Rs {r['Estimated (Rs/ha)']:,}")
            st.write(f"Mehnat: {r['Effort']} | Performance: {r['Performance']}")

    st.subheader("Top 5 fasal: estimated profit/loss")
    top = [x for x in st.session_state.rows if x["Estimated (Rs/ha)"] is not None][:5]
    st.bar_chart(pd.DataFrame(top).set_index("Crop")["Estimated (Rs/ha)"])
    st.caption("Zyada profit wali fasal (Banana, Ginger, Turmeric) mein investment aur risk bhi zyada hota hai.")
