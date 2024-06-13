#############
# BING Images
from tqdm import tqdm
import os
import pandas as pd

# Step 1: Get the list of filenames from the "images" folder
folder_path = 'images'
filenames = os.listdir(folder_path)
row_ids = set()
id2filename = {}

for filename in filenames:
    if '_x4_cropped' in filename:
        bubble_id = filename.split('_x4_cropped')[0]
        bubble_id = bubble_id[:-2]  # drop out the last two digits here
        row_id = int(bubble_id, 4) # convert to decimal
    else:
        print(f"invalid format: {filename}")
        continue
    row_ids.add(int(row_id))
    id2filename[int(row_id)] = filename

print(f"Read {len(row_ids)} lines.")
# Step 2: Define a function to process CSV files and extract matching rows
def extract_matching_rows(csv_file, row_id):
    df = pd.read_csv(csv_file)
    return df[df['id'].isin(row_id)]

# List of CSV files
csv_files = ['Gnyc_wd_evening.csv', 'Gnyc_wd_noon.csv', 'Gnyc_we_evening.csv', 'Gnyc_we_noon.csv']
result_dfs = []

# Step 3: Process each CSV file
for csv_file in tqdm(csv_files, total=len(csv_files)):
    matching_rows = extract_matching_rows(csv_file, row_ids)
    matching_rows["filename"] = matching_rows.id.map(id2filename)
    result_dfs.append(matching_rows)

# Concatenate all the result dataframes
result_df = pd.concat(result_dfs)

# Step 4: Ensure all filenames have a corresponding row
missing_panoids = row_ids - set(result_df['id'])

if missing_panoids:
    print(f"Total: {len(row_ids)}, Missing: {len(missing_panoids)}")
else:
    print("All filenames have corresponding rows in the CSV files.")

# Save the result to a new CSV file if needed
result_df.to_csv('matched_rows.csv', index=False)
