#load libraries
from lib.physicochemical_properties import physicochemical_encoder
from lib.fft_encoding import fft_encoding
import pandas as pd
import sys

#read raw data and encoders
input_data = pd.read_csv(sys.argv[1])
encoders = pd.read_csv(sys.argv[2])
encoders = encoders.fillna(encoders.mean())
encoders.index = encoders['residue']

#define variables
column_with_seq = "sequence"
column_with_id = "target"

#run process
path_export = sys.argv[3]
for prop_value in encoders.columns:

    if prop_value != "residue":
        print("Processing property: ", prop_value)

        encoding_instance = physicochemical_encoder(input_data,
                    prop_value,
                    encoders,
                    column_with_seq,
                    column_with_id)
        
        df_data = encoding_instance.encoding_dataset()

        df_data.to_csv("{}{}.csv".format(path_export, prop_value), index=False)

        fft_encoding_instance = fft_encoding(df_data,
                                        len(df_data.columns)-1,
                                        column_with_id)

        df_data_2 = fft_encoding_instance.encoding_dataset()

        df_data_2.to_csv("{}{}_FFT.csv".format(path_export, prop_value), index=False)