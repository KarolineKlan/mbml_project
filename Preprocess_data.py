import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import csv
import pickle
import numpy as np
import os

def preprocess_df(df):
    # Remove duplicates based on 'Patientkontakt ID' and 'Kontakt varighed (timer)'
    df = df.drop_duplicates(subset=["Patientkontakt ID", "Kontakt varighed (timer)"], keep="first")

    # Remove all Akut contacts
    #df = df[df['Indlæggelsesmåde'] != 'Akut']

    # Group by Patient ID and count unique Aktionsdiagnosekode
    unique_counts = df.groupby('Patient ID')['Aktionsdiagnosekode'].nunique()
    # Map the counts back to the original DataFrame
    df['UniqueCodeCount'] = df['Patient ID'].map(unique_counts)

    # For the column "Patient civilstand" combine all that are not "Gift", "Fraskilt", "Ugift" into one category called "Andet"
    valid_categories = ["Gift", "Fraskilt", "Ugift"]
    df["Patient civilstand"] = df["Patient civilstand"].cat.add_categories("Andet")
    df["Patient civilstand"] = df["Patient civilstand"].apply(lambda x: x if x in valid_categories else "Andet")

    # Embed the diagnosis codes
    with open("Embedding/diagnosis_code_embeddings.pkl", "rb") as f:
        code_embeddings = pickle.load(f)
    
    # Ensure the keys in code_embeddings are strings
    code_embeddings = {str(k): tuple(v) for k, v in code_embeddings.items()}

    # Drop rows with NaN in 'Aktionsdiagnosekode'
    df = df.dropna(subset=["Aktionsdiagnosekode"])

    df["embedded"] = df["Aktionsdiagnosekode"].map(code_embeddings)

    df = df[df["embedded"].notnull()]

    return df

def sum_preprocessed_df(df):
    # Group by 'Patient ID' and 'Aktionsdiagnosekode' and calculate the total time spent in the hospital
    truncated_df = (
        df.groupby(['Patient ID', 'Aktionsdiagnosekode'], observed=True)
        .agg(
            totalDiagnoseKontaktVarighed=('Kontakt varighed (timer)', 'sum'),
            antalKontakter=('Patientkontakt ID', 'count'),
            antalDiagnoser=('UniqueCodeCount', 'first'),
            alder=('Patient alder på kontaktstart tidspunkt', 'mean'),
            gender=('Patient køn', 'first'),
            civilStand=('Patient civilstand', 'first'),
            distanceToHospitalKM=('Distance to Hospital (km)', 'first'),
            embedding=('embedded', 'first'),
        ).reset_index()
    )
    return truncated_df

def create_preprocessed_df(force=False):
    if not os.path.exists("data/CaseRigshospitalet_preprocessed.parquet"):
        if not os.path.exists("data/CaseRigshospitalet_optimized_withDistance.parquet"):
            raise FileNotFoundError("The file 'CaseRigshospitalet_optimized_withDistance.parquet' does not exist in the 'data' directory.")
        else:
            df = pd.read_parquet("data/CaseRigshospitalet_optimized_withDistance.parquet")
            prepr_df = preprocess_df(df)

            # Convert to PyArrow Table and save as Parquet
            table = pa.Table.from_pandas(prepr_df)
            pq.write_table(table, "data/CaseRigshospitalet_preprocessed.parquet")
            print("Preprocessing complete.")
    else:
        if force:
            if not os.path.exists("data/CaseRigshospitalet_optimized_withDistance.parquet"):
                raise FileNotFoundError("The file 'CaseRigshospitalet_optimized_withDistance.parquet' does not exist in the 'data' directory.")
            else:
                df = pd.read_parquet("data/CaseRigshospitalet_optimized_withDistance.parquet")
                prepr_df = preprocess_df(df)

                # Convert to PyArrow Table and save as Parquet
                table = pa.Table.from_pandas(prepr_df)
                pq.write_table(table, "data/CaseRigshospitalet_preprocessed.parquet")
                print("Preprocessing complete.")

        else:
            print("Preprocessed DataFrame already exists. Use force=True to overwrite.")
    return


def create_summed_df(force=False):
    if not os.path.exists("data/CaseRigshospitalet_summed.parquet"):
        create_preprocessed_df(force=force)
        df = pd.read_parquet("data/CaseRigshospitalet_preprocessed.parquet")
        truncated_df = sum_preprocessed_df(df)
        table = pa.Table.from_pandas(truncated_df)
        pq.write_table(table, "data/CaseRigshospitalet_summed.parquet")
        print("Summing and truncating dataset complete.")
    else:
        if force:
            create_preprocessed_df(force=force)
            df = pd.read_parquet("data/CaseRigshospitalet_preprocessed.parquet")
            truncated_df = sum_preprocessed_df(df)
            table = pa.Table.from_pandas(truncated_df)
            pq.write_table(table, "data/CaseRigshospitalet_summed.parquet") 
            print("Summing and truncating dataset complete.")
        else:
            print("Summed DataFrame already exists. Use force=True to overwrite.")
    return



if __name__ == "__main__":
    create_summed_df(force=True)