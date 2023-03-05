import pandas as pd
import numpy as np


def interpolate_nans(x: pd.Series, **kwargs):
    """
    Interpolate a pandas series.


    Arguments
    ---------

    - x: pd.Series:
        The series to interpolate. The index
        will be taken as the x values and the
        values will be taken as the y values.

    - kwargs:
        Keyword arguments that are passed to
        the :code:`np.interp` function.

    Returns
    ---------

    - res: pd.Series:
        The interpolated pandas data series.


    """
    index = x.index
    is_nan = pd.isna(x.values)
    res = x.values * 1.0
    res[is_nan] = np.interp(
        x.index[is_nan], x.index[~is_nan], x.values[~is_nan], **kwargs
    )
    res = pd.Series(data=res, index=index)
    return res
