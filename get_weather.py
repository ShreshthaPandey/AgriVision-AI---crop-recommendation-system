
import pandas as pd, numpy as np, requests, time, os

d = pd.read_csv("district_latlon_final.csv")
OUT = "weather_partial.csv"

done = set()
rows = []
if os.path.exists(OUT):
    old = pd.read_csv(OUT)
    rows = old.to_dict("records")
    done = set(zip(old["State"], old["District"]))

def season_of(m):
    if m in (6, 7, 8, 9, 10): return "Kharif"
    if m in (11, 12, 1, 2, 3): return "Rabi"
    return "Zaid"

for i, r in d.iterrows():
    key = (r["State"], r["District"])
    if key in done:
        continue
    try:
        resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": r["lat"], "longitude": r["lon"],
            "start_date": "2017-01-01", "end_date": "2019-12-31",
            "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
            "timezone": "Asia/Kolkata"}, timeout=60)
        w = pd.DataFrame(resp.json()["daily"])
        w["time"] = pd.to_datetime(w["time"])
        w["Year"] = w["time"].dt.year
        w["Month"] = w["time"].dt.month
        w["Season"] = w["Month"].map(season_of)

        # Season-wise: har saal ka total rain, phir 3 saal ka average
        sy = w.groupby(["Year", "Season"]).agg(
            Rain=("precipitation_sum", "sum"),
            Temp=("temperature_2m_mean", "mean"),
            Humidity=("relative_humidity_2m_mean", "mean")).reset_index()
        avg = sy.groupby("Season")[["Rain", "Temp", "Humidity"]].mean().reset_index()

        # Whole Year
        wy = w.groupby("Year").agg(
            Rain=("precipitation_sum", "sum"),
            Temp=("temperature_2m_mean", "mean"),
            Humidity=("relative_humidity_2m_mean", "mean")).mean()
        avg = pd.concat([avg, pd.DataFrame([{"Season": "Whole Year",
                         "Rain": wy["Rain"], "Temp": wy["Temp"], "Humidity": wy["Humidity"]}])])

        for _, a in avg.iterrows():
            rows.append({"State": r["State"], "District": r["District"],
                         "Season": a["Season"], "Rain": a["Rain"],
                         "Temp": a["Temp"], "Humidity": a["Humidity"]})
        done.add(key)
    except Exception as e:
        print("error", key, str(e)[:80], flush=True)
        time.sleep(30)
        continue
    time.sleep(3)
    if len(done) % 10 == 0:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(len(done), "districts done", flush=True)

pd.DataFrame(rows).to_csv("weather_final.csv", index=False)
print("Finished:", len(done), "districts")
