import sys
import os

path_input = sys.argv[1]
path_save_results = sys.argv[2]
number_epochs = sys.argv[3]

folders = ['coding', 'coding_FFT', 'coding_properties']

for folder in folders:
    list_document = os.listdir("{}{}".format(path_input, folder))
    for element in list_document:
        print("Processing document: ", element)

        command = "python cnn_explorer.py {}{}/{} {} {} {}_{}".format(
            path_input,
            folder,
            element,
            number_epochs,
            path_save_results,
            folder,
            element.split(".")[0]
        )

        print(command)
        os.system(command)
        