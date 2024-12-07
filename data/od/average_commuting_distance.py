import pandas as pd
import numpy as np

"""
Calculate the average commuting distance by commune
"""

def configure(context):
    context.stage("data.od.cleaned")
    context.stage("data.spatial.codes")
    context.stage("data.spatial.centroid_distances")
    context.config("output_path")
    context.config("output_prefix", "ile_de_france_")

def fix_origins(df, commune_ids): 
    existing_ids = set(np.unique(df["origin_id"]))
    missing_ids = commune_ids - existing_ids

    rows = []
    for origin_id in missing_ids:
        for destination_id in commune_ids:
            rows.append((origin_id, destination_id, 1.0 if origin_id == destination_id else 0.0))

    print("Fixing %d origins for commute" % (len(missing_ids)))
    return pd.concat([df, pd.DataFrame.from_records(
        rows, columns = ["origin_id", "destination_id", "weight"]
    )]).sort_values(["origin_id", "destination_id"])

def execute(context):
    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")

    # Load and clean data
    df_work, df_education = context.stage("data.od.cleaned")
    df_distances = context.stage("data.spatial.centroid_distances")
    df_codes = context.stage("data.spatial.codes")
    commune_ids = set(df_codes["commune_id"].unique())

    # Aggregate the flows
    df_commute = pd.concat([df_work, df_education],axis=0)
    df_commute = df_commute.groupby(["origin_id", "destination_id"])["weight"].sum().reset_index()
    df_commute = fix_origins(df_commute, commune_ids)

    # Add centroid and cumulated distances information
    df_commute = pd.merge(left=df_commute, right=df_distances, how="left", on=["origin_id", "destination_id"])
    df_commute["cumulated_commuting_distance"] = df_commute.apply(lambda row: row["centroid_distance"]*row["weight"], axis=1)

    # Calculate mean commuting distance by commune
    df_total = df_commute.groupby(["origin_id"])["weight"].sum().reset_index().rename({"weight":"total_flow"}, axis = 1)
    df_commute = df_commute.groupby(["origin_id"])["cumulated_commuting_distance"].sum().reset_index()
    df_commute = pd.merge(left=df_commute, right=df_total, how="left", on = "origin_id")
    df_commute["mean_commuting_distance"] = df_commute.apply(lambda row: row["cumulated_commuting_distance"] / row["total_flow"], axis=1)

    # Clean output df
    df_commute = df_pt[["origin_id","mean_commuting_distance"]].rename(columns={"origin_id":"commune"})
    df_commute.to_csv("%s/%mean_commuting_distance.csv" % (output_path, output_prefix), sep=";", index=None)
    return df_commute
