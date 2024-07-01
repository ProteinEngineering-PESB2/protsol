import pandas as pd
import sys
import os

from physicochemical_properties import PhysicochemicalEncoder

path_input = sys.argv[1]
name_activity = sys.argv[2]
name_encoders = sys.argv[3]

print("Reading datasets")
df_data_training = pd.read_csv(f"{path_input}train_dataset.csv")
df_data_testing = pd.read_csv(f"{path_input}test_dataset.csv")

list_dfs = [
    (df_data_training, "training_dataset"),
    (df_data_testing, "testing_dataset"),
]

for group in range(8):
    group_to_process = f"Group_{group}"
    command = f"mkdir -p {path_input}{group_to_process}"
    print(command)
    os.system(command)

    print("Start codifications")

    for element in list_dfs:

        print("Processing df: ", element[1])

        df_data = element[0]
        name_export = f"{path_input}{group_to_process}/{element[1]}.csv"

        physicochemical_encoder = PhysicochemicalEncoder(
            dataset=df_data,
            dataset_encoder=pd.read_csv(name_encoders),
            columns_to_ignore=[name_activity],
            name_column_seq="sequence"
        )

        physicochemical_encoder.run_process()

        physicochemical_encoder.df_data_encoded.to_csv(name_export, index=False)