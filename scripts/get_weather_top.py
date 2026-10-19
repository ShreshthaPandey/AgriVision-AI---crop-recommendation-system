
import pandas as pd, numpy as np, requests, time, os

d = pd.read_csv("top_districts.csv")
OUT = "weather_top_partial.csv"

rows, done = [], set()
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
    ok = False
    for attempt in range(4):
        try:
            resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": r["lat"], "longitude": r["lon"],
                "start_date": "2017-01-01", "end_date": "2019-12-31",
                "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
                "timezone": "Asia/Kolkata"}, timeout=60)
            w = pd.DataFrame(resp.json()["daily"])
            ok = True
            break
        except Exception as e:
            print("retry", key, attempt, flush=True)
            time.sleep(20)
    if not ok:
        continue
    w["time"] = pd.to_datetime(w["time"])
    w["Year"] = w["time"].dt.year
    w["Season"] = w["time"].dt.month.map(season_of)
    sy = w.groupby(["Year", "Season"]).agg(
        Rain=("precipitation_sum", "sum"),
        Temp=("temperature_2m_mean", "mean"),
        Humidity=("relative_humidity_2m_mean", "mean")).reset_index()
    avg = sy.groupby("Season")[["Rain", "Temp", "Humidity"]].mean().reset_index()
    wy = w.groupby("Year").agg(
        Rain=("precipitation_sum", "sum"),
        Temp=("temperature_2m_mean", "mean"),
        Humidity=("relative_humidity_2m_mean", "mean")).mean()
    avg = pd.concat([avg, pd.DataFrame([{"Season": "Whole Year", "Rain": wy["Rain"],
                     "Temp": wy["Temp"], "Humidity": wy["Humidity"]}])])
    for _, a in avg.iterrows():
        rows.append({"State": r["State"], "District": r["District"], "Season": a["Season"],
                     "Rain": a["Rain"], "Temp": a["Temp"], "Humidity": a["Humidity"]})
    done.add(key)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(len(done), "done", flush=True)
    time.sleep(4)

pd.DataFrame(rows).to_csv(OUT, index=False)
print("Finished:", len(done))
