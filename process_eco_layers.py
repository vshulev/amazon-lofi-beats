import pandas as pd
import numpy as np
import rasterio
from rasterio.sample import sample_gen
from tqdm import tqdm

data_path = "amazon_data.tsv"

tif_dir = "/Users/vshulev/Downloads/ecolayers"
tif_files = [
    "median_elevation_1km.tiff",
    "human_footprint.tiff",
    "population_density_1km.tif",
    "annual_precipitation.tif",
    "precipitation_seasonality.tif",
    "annual_mean_air_temp.tif",
    "temp_seasonality.tif",
]

df = pd.read_csv(data_path, sep="\t")
df["embeddings"] = df["embeddings"].apply(lambda x: np.array(list(map(float, x[1:-1].split()))))

def parse_coords(df):
    coords = []
    for coord in df["coord"]:
        parts = coord[1:-1].split(",")
        x, y = float(parts[0]), float(parts[1])

        coords.append((y, x))

    return coords

def process_ecolayer(ecolayer, coords):
    with rasterio.open(ecolayer) as dataset:
        # Get the corresponding ecological values for the samples
        results = sample_gen(dataset, coords)
        results = [r for r in results]
    return results

coords = parse_coords(df)

for tif_file in tqdm(tif_files):
    results = process_ecolayer(f"{tif_dir}/{tif_files[0]}", coords)
    df[tif_file.replace(".tiff", "").replace(".tif", "")] = [np.mean(r) for r in results]

print(df.head())
df.to_csv("amazon_data_with_ecolayers.tsv", sep="\t", index=False)
