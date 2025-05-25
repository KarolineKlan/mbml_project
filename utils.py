import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def summarize_columns(df):
    excluded_column = "embedding"
    columns_to_summarize = [c for c in df.columns if c != excluded_column]
    
    print(pd.DataFrame([
        (
            c,
            df[c].dtype,
            len(df[c].unique()),
            df[c].memory_usage(deep=True) // (1024**2)
        ) for c in columns_to_summarize
    ], columns=['name', 'dtype', 'unique', 'size (MB)']))
    
    print('Total size (excluding "embedding"):', 
          df[columns_to_summarize].memory_usage(deep=True).sum() / 1024**2, 'MB')


def make_demographic_features(df):
    # Create dummy variables for categorical features
    df = pd.get_dummies(df, columns=["gender", "civilStand"], drop_first=True)
    
    # Ensure all columns in demographic_columns are numeric
    demographic_columns = ["alder", "distanceToHospitalKM"] + [col for col in df.columns if col.startswith("gender_") or col.startswith("civilStand_")]

    # Check for non-numeric columns
    for col in demographic_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Column {col} is not numeric. Converting to numeric.")
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric, setting invalid values to NaN

    # Handle missing values (e.g., fill NaN with 0)
    df[demographic_columns] = df[demographic_columns].fillna(0)

    print("Demographic columns:", demographic_columns)
    
    return df[demographic_columns].values.astype(np.float32)

def split_patient_level(df, total_samples=100000):
    # Convert list of lists to a 2D NumPy array
    embedding_length = len(df["embedding"].iloc[0])  # Assuming all embeddings should have the same length
    df["embedding"] = df["embedding"].apply(lambda x: x if len(x) == embedding_length else [0.0] * embedding_length)
    embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)
    embedding_tensor = torch.from_numpy(embeddings)

    # Make vector for each row with alder, gender (one hot encoding), civilStand (one hot encoding), distanceToHospitalKM
    demographic_data = torch.from_numpy(make_demographic_features(df))

    # 1. Convert target to numpy
    y_time = torch.from_numpy(df['totalDiagnoseKontaktVarighed'].values.astype(np.float32))
    y_count = torch.from_numpy(df["antalKontakter"].values.astype(np.int16))

    # 2. Randomly select indices
    if total_samples:
        if total_samples > len(df):

            print("total_samples cannot be greater than the number of rows in the DataFrame. Adjusting to the maximum available samples.")
            total_samples = len(df)
            selected_indices = np.arange(len(embedding_tensor))
            df_subset = df.copu()
        else:
            all_indices = np.arange(len(embedding_tensor))

            selected_indices = np.random.choice(all_indices, size=total_samples, replace=False)
            df_subset = df.iloc[selected_indices]
    else:
        selected_indices = np.arange(len(embedding_tensor))
        df_subset = df.copy()

    # Select based on patient ID, can't have same patient in train and test
    unique_patient_ids = df_subset['Patient ID'].unique()

    train_patient_ids, test_patient_ids = train_test_split(
        unique_patient_ids, test_size=0.2, random_state=42
    )
    train_mask = df_subset['Patient ID'].isin(train_patient_ids)
    test_mask = df_subset['Patient ID'].isin(test_patient_ids)

    # Use the mask to filter rows
    x_emb_train = embedding_tensor[selected_indices][train_mask.values]
    d_demo_train = demographic_data[selected_indices][train_mask.values]
    v_time_train = y_time[selected_indices][train_mask.values]
    a_count_train = y_count[selected_indices][train_mask.values]

    x_emb_test = embedding_tensor[selected_indices][test_mask.values]
    d_demo_test = demographic_data[selected_indices][test_mask.values]
    v_time_test = y_time[selected_indices][test_mask.values]
    a_count_test = y_count[selected_indices][test_mask.values]
    return (x_emb_train, d_demo_train, v_time_train, a_count_train), (x_emb_test, d_demo_test, v_time_test, a_count_test)