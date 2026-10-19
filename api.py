from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd, joblib

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = joblib.load("models/final_model.pkl")
d = pd.read_csv("data/processed/final_data.csv")
ref = pd.read_csv("data/processed/crop_reference.csv").set_index("Crop")
sw = pd.read_csv("data/processed/state_weather.csv")

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


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/states")
def states():
    return sorted(d["State"].unique().tolist())


@app.get("/districts")
def districts(state: str):
    return sorted(d[d["State"] == state]["District"].unique().tolist())


@app.get("/seasons")
def seasons(state: str, district: str):
    s = d[(d["State"] == state) & (d["District"] == district)]["Season"]
    return sorted(s.unique().tolist())


@app.get("/weather")
def weather(state: str, season: str, crop: str):
    w = sw[(sw["State"] == state) & (sw["Season"] == season)]
    if w.empty:
        return {"available": False}
    t, r, h = float(w["Temp"].iloc[0]), float(w["Rain"].iloc[0]), float(w["Humidity"].iloc[0])
    res = {"available": True, "temp": round(t, 1), "rain": round(r), "humidity": round(h)}
    if crop in ideal:
        tmin, tmax, rmin = ideal[crop]
        ok = int(tmin <= t <= tmax) + int(r >= rmin)
        res.update({"tmin": tmin, "tmax": tmax, "rmin": rmin,
                    "fit": ["Poor", "Moderate", "Good"][ok]})
    return res


@app.get("/recommend")
def recommend(state: str, district: str, season: str):
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
        out.append({"crop": crop, "yield": round(y, 2), "performance": perf, "effort": effort,
                    "price": price, "cost": cost, "profit": profit,
                    "status": "N/A" if profit is None else ("Profit" if profit > 0 else "Loss"),
                    "confidence": "Low" if crop in low_conf else "OK"})
    out.sort(key=lambda r: (r["profit"] is None, -(r["profit"] or 0)))
    return out
