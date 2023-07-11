import sys
import os

list_input = os.listdir(sys.argv[1])
path_export = sys.argv[2]

for element in list_input:
    command = "python exploring_clf_models.py {}{} {}{}".format(
        sys.argv[1],
        element,
        path_export,
        element
    )

    print(command)
    os.system(command)