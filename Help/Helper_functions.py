import numpy as np
import pandas
import os
from settings.config import PATH_TO_PLOT
import matplotlib.pyplot as plt
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs. -> Numpy arrays

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    >> y = np.array([1, np.nan, 3, np.nan, 5])
    >> nans, index = nan_helper(y)
    >> print(nans)
    >> [False, True, False, True, False]
    >> print(index(nans))
    >> [1, 3]
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def big_number_removal(value):
    """
    Removes numbers with more than one "."
    :param value:
    :return:
    """
    if isinstance(value, str) and value.count('.') > 1:  # meer dan 1 plot
        print("Warning: incorrect values present")
        return float('nan')  # naar NaN omzetten
    return value


def count_selected_columns(index, total_columns):
    """
    Returns the number of selected columns based on the index type.

    Parameters:
    - index: int, slice, or list defining selected columns
    - total_columns: int, total number of columns in the dataset

    Returns:
    - int: number of selected columns
    """
    assert isinstance(total_columns, int), "invalid total column"
    if isinstance(index, int):
        return 1  #1 kolom
    elif isinstance(index, slice):
        return len(range(*index.indices(total_columns)))  #slice
    elif isinstance(index, list) or isinstance(index, tuple):
        return len(index)  #lijst van kolommen
    elif isinstance(index, range):
        return len(index)
    else:
        raise ValueError("Invalid index type. Use int, slice, or list/tuple.")

def shift(index):
    """
    When seperating the ERT data from dates, the indices need to be shifted '1 to the right'.
    Because the index can be an int, a range or a selection of columns, a seperate function is needed
    :param index:
    :return:
    """
    if isinstance(index, int):
        return [index + 1]  #shift index +1
    elif isinstance(index, range):
        return list(range(index.start + 1, index.stop + 1, index.step))  #shift range met +1
    elif isinstance(index, slice):
        return list(range(index.start + 1, index.stop + 1, index.step))  #shift slice met +1
    elif isinstance(index, list):
        return [i + 1 for i in index]  #shift alle elementen in lijst met +1
    else:
        raise AssertionError ("invalid index")

def standardise(data, index):
    if isinstance(index, int):  # Voor een enkele index moet er naar een 2D dataframe geconvert worden
        return pandas.DataFrame(np.array(data).reshape(-1, 1))
    else:
        return data

def export(title, fig=None, dpi=300, extension="png"):
    assert isinstance(extension, str) and isinstance(title, str), "gelieve een string waarde op te geven"
    if fig is None:
        fig = plt.gcf()
    filename = f"{title}.{extension}"
    fname = os.path.join(PATH_TO_PLOT, filename)
    fig.savefig(fname, dpi=dpi)

