###############
# GOOGLE

from PIL import Image
import os
from streetlevel import streetview
import pandas as pd

matched_rows = pd.read_csv("matched_rows.csv")
for index, row in matched_rows.iterrows():
    panoid = row['panoid']
    lat = row['lat']
    lon = row['lon']

    # pano = streetview.find_panorama(lat, lon)
    pano = streetview.find_panorama_by_id(panoid)
    streetview.download_panorama(pano, f"results/{row.filename}")

    if index == 0:
        for names in os.listdir("images"):
            if row.panoid in names:
                im = Image.open(os.path.join("results", f"{row.filename}"))
                im.show()
                im = Image.open(os.path.join("images", names))
                im.show()
                input("is this okay?")

