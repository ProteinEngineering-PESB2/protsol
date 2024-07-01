#!/usr/bin/bash

# categorical solubility
python /home/dmedina/protsol/src/numerical_representation/run_one_hot.py /home/dmedina/protsol/processing_datasets/categorical_dataset/ response

# categorical mutations
python /home/dmedina/protsol/src/numerical_representation/run_one_hot.py /home/dmedina/protsol/processing_datasets/mutational_dataset/ response

# numerical solubility
python /home/dmedina/protsol/src/numerical_representation/run_one_hot.py /home/dmedina/protsol/processing_datasets/numerical_response/ response

