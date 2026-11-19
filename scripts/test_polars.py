import sys
import polars as pl

df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
out = df.group_by("b").agg(pl.col("a").sum()).sort("b")
print(out)
print("polars OK", pl.__version__)

