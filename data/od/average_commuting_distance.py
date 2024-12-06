import pandas as pd
import numpy as np

"""
Calculate the average commuting distance by commune
"""

def configure(context):
    context.stage("data.od.cleaned")
    context.stage("data.spatial.codes")

def fix_origins(df, commune_ids, purpose,category): 
    existing_ids = set(np.unique(df["origin_id"]))
    missing_ids = commune_ids - existing_ids
    categories = set(np.unique(df[category]))

    rows = []
    for origin_id in missing_ids:
        for destination_id in commune_ids:
            for category_name in categories :
                rows.append((origin_id, destination_id, category_name, 1.0 if origin_id == destination_id else 0.0))

    print("Fixing %d origins for %s" % (len(missing_ids), purpose))

    return pd.concat([df, pd.DataFrame.from_records(
        rows, columns = ["origin_id", "destination_id", category, "weight"]
    )]).sort_values(["origin_id", "destination_id"])
  
# df_work["origin_id", "destination_id", "commute_mode","weight"]
def execute(context):
    # Municipalities list
    df_codes = context.stage("data.spatial.codes")
    commune_ids = set(df_codes["commune_id"].unique())

    # Load and clean data
    df_work, df_education = context.stage("data.od.cleaned")
    df_work = fix_origins(df_work, commune_ids, "work","commute_mode")

    # Aggregate work (we do not consider different modes at the moment)
    df_pt = df_work[df_work.commute_mode == "pt"].reset_index(drop=True)
    df_pt = df_pt[["origin_id", "weight"]].groupby(["origin_id"]).sum().reset_index()

    # Compute total and PT share
    df_total = df_work[["origin_id", "weight"]].groupby("origin_id").sum().reset_index().rename({ "weight" : "total" }, axis = 1)
    df_pt = pd.merge(left=df_pt, right=df_total, how="left", on = "origin_id")
    df_pt["PT_share"] = df_pt.apply(lambda row: 100 * row["weight"] / row["total"], axis=1)

    # Clean output df
    df_pt = df_pt[["origin_id","PT_share"]]
    df_pt = df_pt.rename(columns={"origin_id":"commune"})
    df_pt.to_csv("%s/%PT_share.csv" % (output_path, output_prefix), sep=";", index=None)
    return df_pt
