# tools/inspect_node_predictions_schema.py
import pathlib, pyarrow.dataset as ds, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
NODE_DS = ROOT / "visualizations" / "parquet" / "node_predictions_ds"

def main():
    dataset = ds.dataset(NODE_DS.as_posix(), format="parquet", partitioning="hive")
    print("=== SCHEMA ===")
    print(dataset.schema)

    tbl = dataset.take(range(10))  # 10 dòng đầu
    df = tbl.to_pandas()
    print("\n=== 10 ROWS SAMPLE ===")
    print(df.head(10))

    cols = [c.lower() for c in df.columns]
    print("\n=== COLUMNS (lower) ===")
    print(cols)

    # gợi ý cột geocode
    candidates = [
        "region_code","geocode","code_muni","mun_geocode","municipio_id",
        "nodeid","node_id","id","gid","cd_mun","cd_geocmu","cd_mun_ibge"
    ]
    hits = [c for c in candidates if c in cols]
    print("\n=== Gợi ý cột geocode (khớp tên) ===")
    print(hits or "Không khớp tên nào — xem 10 dòng phía trên để nhận diện.")

if __name__ == "__main__":
    main()
