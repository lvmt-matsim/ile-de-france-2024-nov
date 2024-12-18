from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import numba

import data.hts.egt.cleaned
import data.hts.entd.cleaned

import multiprocessing as mp

"""
This stage fuses census data with HTS data.
"""

def configure(context):
    # population data
    context.stage("synthesis.population.matched")
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.income.selected")
    context.stage("synthesis.population.spatial.locations") # ML
    context.config("output_path") # ML
    context.config("output_prefix", "ile_de_france_") # ML
    context.config("vehicles_method", "default") # ML
    context.config("population_with_mobility_variables", False) # ML
    # HTS data
    hts = context.config("hts")
    context.stage("data.hts.selected", alias = "hts")
    # PT share and commuting distance by commune
    context.stage("data.od.public_transport_share")
    context.stage("data.od.average_commuting_distance")
    context.stage("data.spatial.centroid_distances")

def execute(context):
    output_path = context.config("output_path") # ML
    output_prefix = context.config("output_prefix") # ML
    car_fleet_synthesis_method = context.config("vehicles_method", "default") # ML
    add_mobility_variables = context.config("population_with_mobility_variables") # ML
    if car_fleet_synthesis_method == "household_assignment" : # ML
        add_mobility_variables = True
    
    # Select population columns
    df_population = context.stage("synthesis.population.sampled")[[
        "person_id", "household_id",
        "census_person_id", "census_household_id",
        "age", "sex", "employed", "studies", "age_range"
        "number_of_vehicles", "household_size", "consumption_units", "socioprofessional_class", 
        "housing_type", "household_type", "parking", "building_age" # Added, ML
    ]]

    # Attach matching information
    df_matching = context.stage("synthesis.population.matched")
    df_population = pd.merge(df_population, df_matching, on = "person_id")

    initial_size = len(df_population)
    initial_person_ids = len(df_population["person_id"].unique())
    initial_household_ids = len(df_population["household_id"].unique())

    # Attach person and household attributes from HTS
    df_hts_households, df_hts_persons, _ = context.stage("hts")
    df_hts_persons = df_hts_persons.rename(columns = { "person_id": "hts_id", "household_id": "hts_household_id" })
    df_hts_households = df_hts_households.rename(columns = { "household_id": "hts_household_id" })

    person_hts_columns = ["hts_id", "hts_household_id", "has_license", "has_pt_subscription", "is_passenger"]
    household_hts_columns = ["hts_household_id", "number_of_bikes"]
    if add_mobility_variables : # Added, ML
        person_hts_columns.append("parking_at_workplace")
        household_hts_columns.extend(["ENERGV1_egt", "APMCV1_egt","ENERGV2_egt", "APMCV2_egt","ENERGV3_egt", "APMCV3_egt","ENERGV4_egt", "APMCV4_egt"])
    df_population = pd.merge(df_population, df_hts_persons[person_hts_columns], on = "hts_id")
    df_population = pd.merge(df_population, df_hts_households[household_hts_columns], on = "hts_household_id")

    # Attach commune locations to persons
    df_locations = context.stage("synthesis.population.spatial.locations")[["person_id", "purpose", "commune_id"]] # Added, ML
    df_work_education = df_locations[df_locations.purpose.isin(["work", "education"])].reset_index(drop=True).rename(columns={"commune_id":"work_education_commune"}) # Added, ML
    df_home = df_locations[df_locations.purpose == "home"].reset_index(drop=True) # Added, ML
    df_population = pd.merge(left=df_population, right=df_home["person_id", "commune_id"], how="left", on="person_id") # Added, ML
    df_population = pd.merge(left=df_population, right=df_work_education["person_id", "work_education_commune"], how="left", on="person_id") # Added, ML

    # Check BYIN, to remove !
    df_population.to_csv("%s/%spopulation_test.csv" % (output_path, output_prefix), sep=";", index=None)

    # Attach income
    df_income = context.stage("synthesis.population.income.selected")
    df_population = pd.merge(df_population, df_income[[
        "household_id", "household_income"
    ]], on = "household_id")

    # Add car availability
    df_number_of_cars = df_population[["household_id", "number_of_vehicles"]].drop_duplicates("household_id")
    df_number_of_licenses = df_population[["household_id", "has_license"]].groupby("household_id").sum().reset_index().rename(columns = { "has_license": "number_of_licenses" })
    df_car_availability = pd.merge(df_number_of_cars, df_number_of_licenses)

    df_car_availability["car_availability"] = "all"
    df_car_availability.loc[df_car_availability["number_of_vehicles"] < df_car_availability["number_of_licenses"], "car_availability"] = "some"
    df_car_availability.loc[df_car_availability["number_of_vehicles"] == 0, "car_availability"] = "none"
    df_car_availability["car_availability"] = df_car_availability["car_availability"].astype("category")

    df_population = pd.merge(df_population, df_car_availability[["household_id", "car_availability"]])

    # Add bike availability
    df_population["bike_availability"] = "all"
    df_population.loc[df_population["number_of_bikes"] < df_population["household_size"], "bike_availability"] = "some"
    df_population.loc[df_population["number_of_bikes"] == 0, "bike_availability"] = "none"
    df_population["bike_availability"] = df_population["bike_availability"].astype("category")

    # Add individuals commuting distance  # Added ML
    df_distances = context.stage("data.spatial.centroid_distances").rename(columns={"centroid_distance":"commuting_distance"})
    df_population_commuting = df_population[df_population["work_education_commune"].isnan()==False]
    df_population_commuting = pd.merge(left=df_population_commuting, right=df_distances, how="left", left_on=["commune_id", "work_education_commune"] ,right_on=["origin_id", "destination_id"])
    df_population_no_commute = df_population[(df_population["work_education_commune"].isnan()==True) & (df_population["employed"]==True))]
    df_population_unemployed = df_population[(df_population[("work_education_commune"].isnan()==True) & (df_population["employed"]==False)]
    df_population_unemployed["commuting_distance"] = 0.0
    if add_mobility_variables : # Impute the mean commuting distance within the residence commune if no commuting distance
        df_population_commuting["imputed_commuting_distance"] = False
        df_population_unemployed["imputed_commuting_distance"] = False
        df_commuting_dist = context.stage("data.od.average_commuting_distance")
        df_population_no_commute = pd.merge(left=df_population_no_commute, right=df_commuting_dist, how="left", on="commune_id")
        df_population_no_commute["imputed_commuting_distance"] = True
    df_population = pd.concat([df_population_commuting, df_population_no_commute, df_population_unemployed]).sort_index()

    # Households dataframe
    df_households = df_population.rename(columns = { "household_income": "income" }).drop_duplicates("household_id")
    columns_households = ["household_id", "home_commune", "income", "household_type", "car_availability", "bike_availability", # "household_type" added ML
                           "number_of_vehicles", "number_of_bikes", "census_household_id"]
    
    if add_mobility_variables : # Added ML
        # Add accesssibility proxy (public transport modal share) for home and work/education communes
        df_pt = context.stage("data.od.public_transport_share") #### ML
        df_population = pd.merge(left=df_population, right=df_pt, how="left", left_on="commune_id", right_on="commune").rename(columns={"PT_share":"PT_share_home"})
        df_population = pd.merge(left=df_population, right=df_pt, how="left", left_on="work_education_commune", right_on="commune").rename(columns={"PT_share":"PT_share_work"})
        
        # Select population attributes related to households
        df_households[columns_households].to_csv("%s/%shouseholds_main_characteristics.csv" % (output_path, output_prefix), sep = ";", index = None, lineterminator = "\n")
        columns_households.extend(["parking", "housing_type", "building_age", "PT_share_home"
            "ENERGV1_egt", "APMCV1_egt","ENERGV2_egt", "APMCV2_egt","ENERGV3_egt", "APMCV3_egt","ENERGV4_egt", "APMCV4_egt" # to compare
        ])

        # Add other explanatory attributes for car type choice
        df_households_detailed = df_population.copy()
        df_age = df_households_detailed.groupby(["household_id"])["age"].max()
        df_parking_work = df_households_detailed.groupby(["household_id"])["parking_at_workplace"].max()
        df_commuting = df_households_detailed.loc[df_households_detailed.groupby(["household_id"])["commuting_distance"].max(),["household_id","commuting_distance","PT_share_work"]]
        df_N_workers = df_households_detailed[["household_id","employed"]].astype({"employed":str}).replace({"True": 1, "False": 0})
        df_N_workers = df_N_workers.groupby(["household_id"])["employed"].sum()
        df_N_workers = df_N_workers['employed'].apply(lambda x: str(x) if x<2 else "2+").rename(columns={"employed":"N_workers"})
        df_households = pd.merge([df_households[columns_households], df_age, df_parking_work, df_commuting, df_N_workers], 
                                 how="left", on="household_id") 
    else :
        df_households = df_households[columns_households]

    # Check consistency
    final_size = len(df_population)
    final_person_ids = len(df_population["person_id"].unique())
    final_household_ids = len(df_households["household_id"].unique())
    assert initial_size == final_size
    assert initial_person_ids == final_person_ids
    assert initial_household_ids == final_household_ids
    
    return df_population, df_households
