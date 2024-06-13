##########
# BING

import matplotlib.pyplot as plt
from PIL import Image
import os
from streetlevel import streetside
import pandas as pd
from tqdm import tqdm
from datetime import datetime

matched_rows = pd.read_csv("matched_rows.csv")
for index, row in tqdm(matched_rows.iterrows(), total=len(matched_rows)):
    id_ = row['id']
    lat = row['lat']
    lon = row['lon']
    reference_datetime_str = row.timestamp.split('-')[0]
    reference_datetime = datetime.strptime(reference_datetime_str, '%m/%d/%Y %H:%M:%S')

    panos = streetside.find_panoramas(lat, lon)

    matching_dates = [entry for entry in panos if entry.date == reference_datetime]
    assert len(matching_dates) == 1
    pano = matching_dates[0]
    streetside.download_panorama(pano, f"results/{row.filename}.png")

    # DEBUG: double-check with our local repository that this matches
    if index == 0:
        for names in os.listdir("images"):
            if row.filename in names:
                im = Image.open(os.path.join("results", f"{row.filename}.png"))
                im.show()
                im = Image.open(os.path.join("images", names))
                im.show()
                input("is this okay?")

