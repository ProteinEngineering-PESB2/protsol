import time
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from cnn_architectures_class import Models


df = pd.read_csv(sys.argv[1])
number_epochs = int(sys.argv[2])
path_export = sys.argv[3]
suffix_data = sys.argv[4]

response = df['target']
df_data = df.drop(columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(df_data.values, response, test_size=0.3, random_state=42)

for archiquecture in ["A", "B", "C", "D"]:

    print("Exploring architecture: ", archiquecture)
    time_inicio = time.time()
    models = Models(X_train, y_train, X_test, y_test, archiquecture)
    models.fit_models(number_epochs, 1)
    metrics = models.get_metrics()
    time_fin = time.time()
    delta_time = round(time_fin - time_inicio, 4)

    metrics["total_time"] = delta_time
    metrics["epochs"] = number_epochs

    name_export = "{}{}_{}.json".format(
        path_export,
        suffix_data,
        archiquecture
    )
    with open(name_export, mode = "w", encoding = "utf-8") as file:
        json.dump(metrics, file)
