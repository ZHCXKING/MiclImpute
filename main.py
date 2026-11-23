import pandas as pd
df = pd.read_parquet("data/Adult/DropNaData.parquet")
print(df)
print(df.info())