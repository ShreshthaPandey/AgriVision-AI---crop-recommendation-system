
import subprocess, os
import pandas as pd

for k in range(15):
    subprocess.run(["python", "get_weather.py"])
    if os.path.exists("weather_final.csv"):
        w = pd.read_csv("weather_final.csv")
        if w[["State", "District"]].drop_duplicates().shape[0] >= 720:
            print("Sab complete")
            break
