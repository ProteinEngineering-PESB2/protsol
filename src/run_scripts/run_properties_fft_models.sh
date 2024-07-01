#!/usr/bin/bash

# categorical solubility
python /home/dmedina/protsol/src/numerical_representation/run_physicochemical_encoder.py /home/dmedina/protsol/processing_datasets/categorical_dataset/ response /home/dmedina/protsol/input_data_for_coding/cluster_encoders.csv
python /home/dmedina/protsol/src/numerical_representation/run_fft_encoder.py /home/dmedina/protsol/processing_datasets/categorical_dataset/ response

# categorical mutations
python /home/dmedina/protsol/src/numerical_representation/run_physicochemical_encoder.py /home/dmedina/protsol/processing_datasets/mutational_dataset/ response /home/dmedina/protsol/input_data_for_coding/cluster_encoders.csv
python /home/dmedina/protsol/src/numerical_representation/run_fft_encoder.py /home/dmedina/protsol/processing_datasets/mutational_dataset/ response

# numerical solubility
python /home/dmedina/protsol/src/numerical_representation/run_physicochemical_encoder.py /home/dmedina/protsol/processing_datasets/numerical_response/ response /home/dmedina/protsol/input_data_for_coding/cluster_encoders.csv
python /home/dmedina/protsol/src/numerical_representation/run_fft_encoder.py /home/dmedina/protsol/processing_datasets/numerical_response/ response

