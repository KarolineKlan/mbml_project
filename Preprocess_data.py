import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import csv


table = csv.read_csv("data/Case Rigshospitalet.csv")
df = table.to_pandas()

def optimize_df(df):
    df['Kontakt startdato'] = pd.to_datetime(
    df['Kontakt startdato'].str.replace(',', '.', regex=False), 
    format="%Y-%m-%d %H:%M:%S.%f", 
    errors='coerce'
    )
    df['Kontakt slutdato'] = pd.to_datetime(
        df['Kontakt slutdato'].str.replace(',', '.', regex=False), 
        format="%Y-%m-%d %H:%M:%S.%f", 
        errors='coerce'
    )
    df['Procedure udført'] = pd.to_datetime(
        df['Procedure udført'].str.replace(',', '.', regex=False), 
        format="%Y-%m-%d %H:%M:%S.%f", 
            errors='coerce'
    )
    
    df['Patient ID'] = df['Patient ID'].astype('category')
    df['Patient alder på kontaktstart tidspunkt'] = pd.to_numeric(df['Patient alder på kontaktstart tidspunkt'], errors='coerce').astype('Int16')
    df['Kontakt startdato'] = pd.to_datetime(df['Kontakt startdato'], format='mixed')
    df['Kontakt slutdato'] = pd.to_datetime(df['Kontakt slutdato'], format='mixed')
    df['Kontakttype'] = df['Kontakttype'].astype('category')
    df['Indlæggelsesmåde'] = df['Indlæggelsesmåde'].astype('category')
    df['Patientkontakttype'] = df['Patientkontakttype'].astype('category')
    df['Aktionsdiagnosekode'] = df['Aktionsdiagnosekode'].astype('category')
    df['Bidiagnosekode'] = df['Bidiagnosekode'].astype('category')
    df['Behandlingsansvarlig Afdeling'] = df['Behandlingsansvarlig Afdeling'].astype('category')
    df['Procedure-kode'] = df['Procedure-kode'].astype('category')
    df['Procedure-tillægskoder'] = df['Procedure-tillægskoder'].astype('category')
    df['Procedure udført'] = pd.to_datetime(df['Procedure udført'], format='mixed')
    
    # Ensure conversion runs only if needed
    if df['Kontakt varighed (timer)'].dtype == 'object':  
        df['Kontakt varighed (timer)'] = df['Kontakt varighed (timer)'].str.replace(',', '.').astype('float32')

    df['Besøgstype'] = df['Besøgstype'].astype('category')
    df['Patient køn'] = df['Patient køn'].astype('category')
    df['Patient civilstand'] = df['Patient civilstand'].astype('category')
    df['Patient oprettet på Min SP (J/N)'] = df['Patient oprettet på Min SP (J/N)'].astype('category')
    df['Patient land'] = df['Patient land'].astype('category')
    df['Patient region'] = df['Patient region'].astype('category')
    df['Patient postnummer'] = pd.to_numeric(df['Patient postnummer'], errors='coerce').astype('Int32')
    df['Patient kommune'] = df['Patient kommune'].astype('category')

    return df

def preprocess_df(df):
    # Remove duplicates based on 'Patientkontakt ID' and 'Kontakt varighed (timer)'
    df = df.drop_duplicates(subset=["Patientkontakt ID","Kontakt varighed (timer)"],keep="first")

    # Remove all Akut contacts
    df = df[df['Indlæggelsesmåde'] != 'Akut']

    # Group by Patient ID and count unique Aktionsdiagnosekode
    unique_counts = df.groupby('Patient ID')['Aktionsdiagnosekode'].nunique()
    # Map the counts back to the original DataFrame
    df['UniqueCodeCount'] = df['Patient ID'].map(unique_counts)

    # Embed the diagnosis codes
    with open("Embedding/diagnosis_code_embeddings.pkl", "rb") as f:
        code_embeddings = pickle.load(f)
    df["embedded"] = df["Aktionsdiagnosekode"].map(code_embeddings)

    return df


# Optimize DataFrame
opt_df = optimize_df(df)
opt_df = preprocess_df(opt_df)

# Convert to PyArrow Table and save as Parquet
table = pa.Table.from_pandas(opt_df)
pq.write_table(table, "data/CaseRigshospitalet_optimized.parquet")