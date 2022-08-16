import numpy as np


def countna(array:np.ndarray, normalise:bool=True,)->float:
    '''
    Function to calculate the number of NAs in
    an array.

    Arguments
    ---------

    - `array`: `numpy.ndarray`:
        The array to calculate the number of missing
        values on.
    
    - `normalise`: `bool`, optional:
        Whether to return the values as a percentage.
    
    Returns
    ---------

    - `countna`: `float`:
        A `float` containing the result.

    '''
    count_na = np.count_nonzero(np.isnan(array))
    if normalise:
        count_na *= 1/array.shape[0]
    return count_na



def format_mean_iqr_missing(values, string="{mean:.2f} ({iqr:.2f}) (({count_na:.0f}%))"):
    mean = np.mean(values)
    iqr = np.subtract(*np.nanpercentile(values, [75, 25]))
    count_na = countna(values)*100
    return string.format(mean=mean, iqr=iqr, count_na=count_na)