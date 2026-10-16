import pandas as pd
import time

print("Converting CRSP Daily CSV to Parquet (one-time operation)...")
print("This will take a while but only needs to be done once.\n")

t0 = time.time()

# Read CSV in chunks to avoid memory issues
chunk_size = 1_000_000
chunks = []

print("Reading CSV in chunks...")
for i, chunk in enumerate(pd.read_csv("../data_raw/CRSP Daily Stock File.csv",
                                      chunksize=chunk_size,
                                      low_memory=False), start=1):
    chunks.append(chunk)
    print(f"  Chunk {i}: {len(chunk):,} rows")

    # Save memory - convert and save every 10 chunks
    if i % 10 == 0:
        print(f"  Concatenating and saving intermediate file...")
        temp = pd.concat(chunks, ignore_index=True)
        temp.to_parquet(f"../data_raw/crsp_daily_temp_{i}.parquet", index=False)
        chunks = []  # Clear memory

# Save any remaining chunks
if chunks:
    temp = pd.concat(chunks, ignore_index=True)
    temp.to_parquet(f"../data_raw/crsp_daily_temp_final.parquet", index=False)

t1 = time.time()
print(f"\nCSV reading completed in {(t1 - t0) / 60:.1f} minutes")

# Combine all parquet files
print("\nCombining parquet files...")
import glob

parquet_files = sorted(glob.glob("../data_raw/crsp_daily_temp_*.parquet"))
dfs = [pd.read_parquet(f) for f in parquet_files]
crsp_daily = pd.concat(dfs, ignore_index=True)

# Save final parquet
crsp_daily.to_parquet("../data_raw/CRSP_Daily_Stock_File.parquet", index=False)

t2 = time.time()
print(f"Parquet saved in {(t2 - t1) / 60:.1f} minutes")
print(f"Total time: {(t2 - t0) / 60:.1f} minutes")
print(f"\nFinal dataset: {len(crsp_daily):,} rows")

# Clean up temp files
print("\nCleaning up temporary files...")
import os

for f in parquet_files:
    os.remove(f)