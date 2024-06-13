###############
# GOOGLE

import os
import pandas as pd

folder_path = 'images'
filenames = os.listdir(folder_path)
panoids = set()
panoid2filename = {}

for filename in filenames:
    if '_right' in filename:
        panoid = filename.split('_right')[0]
    elif '_left' in filename:
        panoid = filename.split('_left')[0]
    else:
        print(f"invalid format: {filename}")
        continue
    panoids.add(panoid)
    panoid2filename[panoid] = filename

# Step 2: Define a function to process CSV files and extract matching rows
def extract_matching_rows(csv_file, panoid_set):
    df = pd.read_csv(csv_file)
    return df[df['panoid'].isin(panoid_set)]

# List of CSV files
csv_files = ['G340890_50m_panoid.csv', 'G340911_50m_panoid.csv', 'G340919_50m_panoid.csv', 'G340929_50m_panoid.csv', 'G340931_50m_panoid.csv']
result_dfs = []

# Step 3: Process each CSV file
for csv_file in csv_files:
    matching_rows = extract_matching_rows(csv_file, panoids)
    matching_rows["filename"] = matching_rows.panoid.map(panoid2filename)
    result_dfs.append(matching_rows)

result_df = pd.concat(result_dfs)


missing_panoids = panoids - set(result_df['panoid'])
if missing_panoids:
    print(f"Missing panoids: {missing_panoids}")
else:
    print("All filenames have corresponding rows in the CSV files.")

# Save the result to a new CSV file if needed
result_df.to_csv('matched_rows.csv', index=False)
