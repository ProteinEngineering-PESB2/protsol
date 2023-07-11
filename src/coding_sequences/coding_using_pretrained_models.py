#libraries
import pandas as pd
from lib.apply_pretrained_models import using_bioembedding
import sys

#data to encode
input_data = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]
option = int(sys.argv[3])

#define variables
column_with_seq = "sequence"
column_with_id = "target"

#instance bio embedding
using_bioembedding_instance = using_bioembedding(
    dataset=input_data,
    id_seq=column_with_id,
    column_seq=column_with_seq,
    is_reduced=True,
    device = 'cuda'
)

#running encoders
if option == 1:
    try:
        #### bepler
        print("Apply bepler")
        using_bioembedding_instance.apply_bepler()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}bepler.csv".format(path_export), index=False)
    except:
        print("Error encoding bepler")

elif option == 2:
    try:
        ### cpcprot
        print("Apply cpcprot")
        using_bioembedding_instance.apply_cpcprot()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}cpcprot.csv".format(path_export), index=False)
    except:
        print("Error encoding cpcprot")

elif option == 3:
    try:
        ### esm1b
        print("Apply esm1b")
        using_bioembedding_instance.apply_esm1b()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}esm1b.csv".format(path_export), index=False)
    except:
        print("Error encoding esm1b")

elif option == 4:
    try:
        ### esm1v
        print("Apply esm1v")
        using_bioembedding_instance.apply_esm1v()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}esm1v.csv".format(path_export), index=False)
    except:
        print("Error encoding esm1v")

elif option == 5:
    try:
        ### esme
        print("Apply esme")
        using_bioembedding_instance.apply_esme()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}esme.csv".format(path_export), index=False)
    except:
        print("Error encoding esme")

elif option == 6:
    try:
        ### fasttext
        print("Apply fasttext")
        using_bioembedding_instance.apply_fasttext()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}fasttext.csv".format(path_export), index=False)
    except:
        print("Error encoding fasttext")

elif option == 7:
    try:
        ### glove
        print("Apply glove")
        using_bioembedding_instance.apply_glove()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}glove.csv".format(path_export), index=False)
    except:
        print("Error encoding glove")

elif option == 8:
    try:
        ### onehot
        print("Apply onehot")
        using_bioembedding_instance.apply_onehot()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}onehot.csv".format(path_export), index=False)
    except:
        print("Error encoding onehot")

elif option == 9:
    try:
        ### plusrnn
        print("Apply plusrnn")
        using_bioembedding_instance.apply_plus_rnn()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}plusrnn.csv".format(path_export), index=False)
    except:
        print("Error encoding plusrnn")

elif option == 10:
    try:
        ### prottrans_albert
        print("Apply prottrans_albert")
        using_bioembedding_instance.apply_prottrans_albert()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}prottrans_albert.csv".format(path_export), index=False)
    except:
        print("Error encoding albert")

elif option == 11:
    try:
        ### prottrans_bert
        print("Apply prottrans_bert")
        using_bioembedding_instance.apply_prottrans_bert()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}prottrans_bert.csv".format(path_export), index=False)
    except:
        print("Error encoding bert")

elif option == 12:
    try:
        ### prottrans_t5bdf
        print("Apply prottrans_t5bdf")
        using_bioembedding_instance.apply_prottrans_T5BFD()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}prottrans_t5bdf.csv".format(path_export), index=False)
    except:
        print("Error encoding t5bdf")

elif option == 13:
    try:
        ### prottrans_t5_uniref
        print("Apply prottrans_t5_uniref")
        using_bioembedding_instance.apply_prottrans_T5_UniRef()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}prottrans_t5_uniref.csv".format(path_export), index=False)
    except:
        print("Error encoding t5_uniref")

elif option == 14:
    try:
        ### prottrans_t5_xlu50
        print("Apply prottrans_t5_xlu50")
        using_bioembedding_instance.apply_prottrans_T5_XLU50()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}prottrans_t5_xlu50.csv".format(path_export), index=False)
    except:
        print("Error encoding t5_xlu50")

elif option == 15:
    try:
        ### prottrans_xlnet
        print("Apply prottrans_xlnet")
        using_bioembedding_instance.apply_prottrans_XLNetUniRef()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}prottrans_xlnet.csv".format(path_export), index=False)
    except:
        print("Error encoding xlnet")

elif option == 16:
    try:
        ### seqvec
        print("Apply seqvec")
        using_bioembedding_instance.apply_seqvec()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}seqvec.csv".format(path_export), index=False)
    except:
        print("Error encoding seqvec")

else:
    try:
        ### word2vec
        print("Apply word2vec")
        using_bioembedding_instance.apply_word2vec()

        header = ["p_{}".format(i) for i in range(len(using_bioembedding_instance.np_data[0]))]
        df_data = pd.DataFrame(using_bioembedding_instance.np_data, columns=header)
        df_data[column_with_id] = input_data[column_with_id]

        df_data.to_csv("{}word2vec.csv".format(path_export), index=False)
    except:
        print("Error encoding word2vec")