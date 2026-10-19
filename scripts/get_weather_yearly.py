
import pandas as pd, numpy as np, requests, time, os

d = pd.read_csv("top_districts.csv")
OUT = "weather_yearly_partial.csv"

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
    w = None
    for attempt in range(4):
        try:
            resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": r["lat"], "longitude": r["lon"],
                "start_date": "2008-06-01", "end_date": "2020-05-31",
                "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
                "timezone": "Asia/Kolkata"}, timeout=120)
            w = pd.DataFrame(resp.json()["daily"])
            break
        except Exception as e:
            print("retry", key, attempt, flush=True)
            time.sleep(30)
    if w is None:
        continue
    w["time"] = pd.to_datetime(w["time"])
    w["Crop_Year"] = np.where(w["time"].dt.month >= 6, w["time"].dt.year, w["time"].dt.year - 1)
    w["Season"] = w["time"].dt.month.map(season_of)
    agg = dict(Rain=("precipitation_sum", "sum"), Temp=("temperature_2m_mean", "mean"),
               Humidity=("relative_humidity_2m_mean", "mean"))
    a = w.groupby(["Crop_Year", "Season"]).agg(**agg).reset_index()
    b = w.groupby("Crop_Year").agg(**agg).reset_index()
    b["Season"] = "Whole Year"
    x = pd.concat([a, b])
    x.insert(0, "District", r["District"])
    x.insert(0, "State", r["State"])
    rows += x.to_dict("records")
    done.add(key)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(len(done), "done", flush=True)
    time.sleep(5)

pd.DataFrame(rows).to_csv(OUT, index=False)
print("Finished:", len(done))
