'''
This script will create embeddings for diagonosises
'''

import torch
import pandas as pd
import numpy as np

import pickle
from danish_bert_embeddings import DanishBertEmbeddings

def main(save_embeddings=True):
    path = "data/diagnosis.csv"
    df = pd.read_csv(path, sep=';')
    

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    embedder = DanishBertEmbeddings()
    embedder.model.to(device)  

    
    text_col = "Aktionsdiagnosetekst"  
    code_col = "Aktionsdiagnosekode"

    code_embeddings = {}
    embeddings = []

    for idx, row in df.iterrows():
        code = row[code_col]
        text1 = str(row["Aktionsdiagnosetekst"]) if pd.notnull(row["Aktionsdiagnosetekst"]) else ""
        text2 = str(row["Gruppe 1"]) if pd.notnull(row["Gruppe 1"]) else ""
        combined_text = text1 + " " + text2
        embedding = embedder.embed(combined_text, output_numpy=True)



        code_embeddings[code] = embedding

        
        embeddings.append(embedding)

        if idx % 500 == 0:
            print(f"Processed {idx} rows.")

    embeddings = np.array(embeddings)
    
    print(f"Generated embeddings for {len(code_embeddings)} unique codes.")

    if save_embeddings:
        with open("diagnosis_code_embeddings.pkl", "wb") as f:
            pickle.dump(code_embeddings, f)
        print("Embeddings saved to 'diagnosis_code_embeddings.pkl'.")

if __name__ == "__main__":
    main(save_embeddings=True)