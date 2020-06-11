import sys


def get_data_directory_from_cmd_args():

    try:
        directory = sys.argv[1]
    except IndexError:
        raise ValueError("please provide data directory as argument")

    if directory[-1] != "/":
        directory += "/"

    return directory
