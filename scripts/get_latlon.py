import pandas as pd, numpy as np, requests, time, os

def get_latlon(district, state):
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": district, "count": 10, "country_code": "IN"}, timeout=20)
        res = r.json().get("results", [])
        for x in res:
            if state.lower() in str(x.get("admin1", "")).lower():
                return x["latitude"], x["longitude"]
        return (res[0]["latitude"], res[0]["longitude"]) if res else (np.nan, np.nan)
    except Exception:
        return (np.nan, np.nan)

if os.path.exists("district_latlon_partial.csv"):
    districts = pd.read_csv("district_latlon_partial.csv")
else:
    districts = pd.read_csv("districts_list.csv")
    districts["lat"] = np.nan
    districts["lon"] = np.nan

for i in range(len(districts)):
    if pd.notna(districts.at[i, "lat"]):
        continue
    la, lo = get_latlon(districts.at[i, "District"], districts.at[i, "State"])
    districts.at[i, "lat"] = la
    districts.at[i, "lon"] = lo
    time.sleep(0.2)
    if i % 25 == 0:
        districts.to_csv("district_latlon_partial.csv", index=False)
        print(i, "done", flush=True)

districts.to_csv("district_latlon.csv", index=False)
print("Done:", districts["lat"].isna().sum(), "missing")