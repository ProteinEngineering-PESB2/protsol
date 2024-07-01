import pandas as pd
import sys
import os

from embedding_representations import BioEmbeddings

path_input = sys.argv[1]
name_column = sys.argv[2]

print("Reading datasets")
print("Procesing: ", path_input)
df_data_training = pd.read_csv(f"{path_input}train_dataset.csv")
df_data_testing = pd.read_csv(f"{path_input}test_dataset.csv")

list_dfs = [
    (df_data_training, "training_dataset"),
    (df_data_testing, "testing_dataset")
]

print("Start codifications")
for element in list_dfs:

    df_data = element[0]
    
    bioembedding_instance = BioEmbeddings(
        df_data, 
        "sequence", 
        is_reduced=True, 
        device = "cuda",
        path_export=path_input, 
        column_response=name_column
    )

    print("Coding data for: ", element[0])

    bioembedding_instance.apply_esm1b(element[1])
    bioembedding_instance.apply_prottrans_bert(element[1])
    bioembedding_instance.apply_prottrans_albert(element[1])
    bioembedding_instance.apply_prottrans_t5_uniref(element[1])
    bioembedding_instance.apply_prottrans_t5_xlu50(element[1])
    bioembedding_instance.apply_prottrans_t5bdf(element[1])
    bioembedding_instance.apply_prottrans_xlnet(element[1])
    bioembedding_instance.apply_bepler(element[1])
    bioembedding_instance.apply_cpc_prot(element[1])
    bioembedding_instance.apply_fasttextv(element[1])
    bioembedding_instance.apply_glove(element[1])
    bioembedding_instance.apply_plusrnn(element[1])
    bioembedding_instance.apply_seqvec(element[1])
    bioembedding_instance.apply_word2vec(element[1])


