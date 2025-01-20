from tqdm import tqdm
import pandas as pd
import numpy as np
import data.hts.hts as hts

"""
This stage cleans the French population census:
  - Assign new unique integer IDs to households and persons
  - Clean up spatial information and sociodemographic attributes
"""

def configure(context):
    context.stage("data.census.raw")
    context.stage("data.spatial.codes")

    if context.config("use_urban_type", False):
        context.stage("data.spatial.urban_type")

def execute(context):
    df = context.stage("data.census.raw")

    # Construct household IDs for persons with NUMMI != Z
    df_household_ids = df[["CANTVILLE", "NUMMI"]]
    df_household_ids = df_household_ids[df_household_ids["NUMMI"] != "Z"]
    df_household_ids["temporary"] = df_household_ids["CANTVILLE"] + df_household_ids["NUMMI"]
    df_household_ids = df_household_ids.drop_duplicates("temporary")
    df_household_ids["household_id"] = np.arange(len(df_household_ids))
    df = pd.merge(df, df_household_ids, on = ["CANTVILLE", "NUMMI"], how = "left")

    # Fill up undefined household ids (those where NUMMI == Z)
    f = np.isnan(df["household_id"])
    df.loc[f, "household_id"] = np.arange(np.count_nonzero(f)) + df["household_id"].max() + 1
    df["household_id"] = df["household_id"].astype(int)

    # Put person IDs
    df["person_id"] = np.arange(len(df))

    # Sorting
    df = df.sort_values(by = ["household_id", "person_id"])

    # Spatial information
    df["departement_id"] = df["DEPT"].astype("category")

    df["commune_id"] = df["IRIS"].str[:5]
    f_undefined = df["commune_id"].str.contains("Z")
    df.loc[f_undefined, "commune_id"] = "undefined"
    df["commune_id"] = df["commune_id"].astype("category")

    df["iris_id"] = df["IRIS"]
    f_undefined = df["iris_id"].str.contains("Z") | df["iris_id"].str.contains("X")
    df.loc[f_undefined, "iris_id"] = "undefined"
    df["iris_id"] = df["iris_id"].astype("category")

    # Age
    df["age"] = df["AGED"].apply(lambda x: "0" if x == "000" else x.lstrip("0")).astype(int)

    # Add age range for education
    df["age_range"] = "higher_education"
    df[df["age"]<=10,"age_range"] = "primary_school"
    df[df["age"].between(11,14),"age_range"] = "middle_school"
    df[df["age"].between(15,17),"age_range"] = "high_school"
    df["age_range"] = df["age_range"].astype("category")

    # Clean COUPLE
    df["couple"] = df["COUPLE"] == "1"

    # Clean TRANS
    df.loc[df["TRANS"] == "1", "commute_mode"] = np.nan
    df.loc[df["TRANS"] == "2", "commute_mode"] = "walk"
    df.loc[df["TRANS"] == "3", "commute_mode"] = "bike"
    df.loc[df["TRANS"] == "4", "commute_mode"] = "car"
    df.loc[df["TRANS"] == "5", "commute_mode"] = "car"
    df.loc[df["TRANS"] == "6", "commute_mode"] = "pt"
    df.loc[df["TRANS"] == "Z", "commute_mode"] = np.nan
    df["commute_mode"] = df["commute_mode"].astype("category")

    # Weight
    df["weight"] = df["IPONDI"].astype(float)

    # Clean SEXE
    df.loc[df["SEXE"] == "1", "sex"] = "male"
    df.loc[df["SEXE"] == "2", "sex"] = "female"
    df["sex"] = df["sex"].astype("category")

    # Clean employment
    df["employed"] = df["TACT"] == "11"

    # Studies
    df["studies"] = df["ETUD"] == "1"

    # Number of vehicles
    df["number_of_vehicles"] = df["VOIT"].apply(
        lambda x: str(x).replace("Z", "0").replace("X", "0")
    ).astype(int)

    df["number_of_vehicles"] += df["DEROU"].apply(
        lambda x: str(x).replace("U", "0").replace("Z", "0").replace("X", "0")
    ).astype(int)

    # Household size
    df_size = df[["household_id"]].groupby("household_id").size().reset_index(name = "household_size")
    df = pd.merge(df, df_size)

    # Socioprofessional category
    df["socioprofessional_class"] = df["CS1"].astype(int)

    # Consumption units
    df = pd.merge(df, hts.calculate_consumption_units(df), on = "household_id")

    # Housing type, ML
    df.loc[df["TYPL"] == "1", "housing_type"] = "house"
    df.loc[df["TYPL"] == "2", "housing_type"] = "flat"
    df.loc[df["TYPL"].isin(["3","4","5","6","Z"]), "housing_type"] = "others"
    df["housing_type"] = df["housing_type"].astype("category")
  
    # Household type, ML
    df.loc[df["SFM"] == "11", "household_type"] = "SM"
    df.loc[df["SFM"] == "12", "household_type"] = "SW"
    df.loc[df["SFM"] == "21", "household_type"] = "MFF"
    df.loc[df["SFM"] == "22", "household_type"] = "MFM"
    df.loc[df["SFM"].isin(["30"]), "household_type"] = "CWOC"
    df.loc[df["SFM"].isin(["31","32","33","34"]), "household_type"] = "CWC"
    df.loc[df["SFM"].isin(["40","51","52","53","54","61","62","70","ZZ"]), "household_type"] = "others"
    df["household_type"] = df["household_type"].astype("category")
  
    # Parking, ML
    df.loc[df["GARL"] == "1", "parking"] = 1
    df.loc[df["GARL"].isin(["2","Z"]), "parking"] = 0
    df["parking"] = df["parking"].astype(np.int)
  
    # Building construction data, ML
    df["building_age"] = df["ACHLR"].astype("category")

    df = df[[
        "person_id", "household_id", "weight",
        "iris_id", "commune_id", "departement_id",
        "age", "sex", "couple", "age_range",
        "commute_mode", "employed",
        "studies", "number_of_vehicles", "household_size",
        "consumption_units", "socioprofessional_class",
        "housing_type", "household_type", "parking", "building_age" # Added, ML
    ]]

    if context.config("use_urban_type"):
        df_urban_type = context.stage("data.spatial.urban_type")[[
            "commune_id", "urban_type"
        ]]
        
        # Impute urban type
        df = pd.merge(df, df_urban_type, on = "commune_id", how = "left")
        df.loc[df["commune_id"] == "undefined", "urban_type"] = "none"
        df["commune_id"] = df["commune_id"].astype("category")
        assert ~np.any(df["urban_type"].isna()) 

    return df
