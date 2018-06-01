""" CRPM Dataset convention and methods

    All data must be numeric.
"""

def load_dataset(file):
    """ Convert csv file to ndarray, skipping headers on first row.

    Data in csv file is stored as observations in rows with columns representing
    observation variables. The output dataset is inverted with observations in
    columns with variables in rows.
    """
    import csv
    import numpy as np

    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        keys = list(next(reader)) #get headers
        data = np.array(list(reader)).astype("float")
    return keys, data.T
