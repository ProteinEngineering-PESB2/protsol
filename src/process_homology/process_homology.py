import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os
from Bio import SeqIO
import random
from sklearn.model_selection import train_test_split
import sys

def read_fasta(name_input):
    matrix_data = []

    with open(name_input) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            row = [
                record.id,
                str(record.seq)
            ]
            matrix_data.append(row)
    
    df_data = pd.DataFrame(data=matrix_data, columns=["seq_id", "sequence"])
    return df_data

def df_to_fasta(df_data, name_fasta):
    doc_export = open(name_fasta, 'w')

    for index in df_data.index:
        sequence = df_data["sequence"][index]

        doc_export.write(f">{index}\n")
        doc_export.write(f"{sequence}\n")
    
    doc_export.close()

name_df = sys.argv[1]
path_export = sys.argv[2]
ration_benchmark = float(sys.argv[3])
name_column_response = sys.argv[4]
proportion_pos = float(sys.argv[5])
proportion_neg = float(sys.argv[6])

df_data = pd.read_csv(name_df)
df_data_positive = df_data[df_data[name_column_response] == 1]
df_data_negative = df_data[df_data[name_column_response] == 0]

df_to_fasta(df_data_positive, f"{path_export}positive_data.fasta")
df_to_fasta(df_data_negative, f"{path_export}negative_data.fasta")

command_pos = f"cd-hit -i {path_export}positive_data.fasta -o {path_export}positive_data_filter.fasta -c {proportion_pos}"
command_neg = f"cd-hit -i {path_export}negative_data.fasta -o {path_export}negative_data_filter.fasta -c {proportion_neg}"

os.system(command_pos)
os.system(command_neg)

df_positive_filter = read_fasta(f"{path_export}positive_data_filter.fasta")
df_negative_filter = read_fasta(f"{path_export}negative_data_filter.fasta")

print("Positive: ", len(df_positive_filter))
print("Negative: ", len(df_negative_filter))

command = f"rm {path_export}*.fasta"
os.system(command)

command = f"rm {path_export}*.clstr"
os.system(command)

positive_sequences = df_positive_filter["sequence"].tolist()
random.shuffle(positive_sequences)

negative_sequences = df_negative_filter["sequence"].tolist()
random.shuffle(negative_sequences)

df_data_pos = pd.DataFrame()
df_data_pos["sequence"] = positive_sequences
df_data_pos["activity"] = 1

df_data_neg = pd.DataFrame()
df_data_neg["sequence"] = negative_sequences
df_data_neg["activity"] = 0

df_data = pd.concat([df_data_pos, df_data_neg])

print(df_data["activity"].value_counts())
df_data_train, df_data_test = train_test_split(df_data, test_size=ration_benchmark, random_state=42)

df_data_train.to_csv(f"{path_export}train_dataset.csv", index=False)
df_data_test.to_csv(f"{path_export}test_dataset.csv", index=False)

print(df_data_train["activity"].value_counts())

print(df_data_test["activity"].value_counts())