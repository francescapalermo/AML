import numpy as np


def time_interval_range(max_hour:int, minutes_per_hour:int, minute_interval:int)->np.ndarray:
    '''
    Create an array of a time range. 
    
    Edited from https://stackoverflow.com/a/58052181/19451559.
    
    Examples
    ---------

    The following will create a list of values 

    .. code-block::

        >>> time_interval_range(24, 60, 30)
        [
            '00:00',
            '00:30',
            ...,
            '22:30',
            '23:00',
            '23:30'
            ]

    
    Arguments
    ---------
    
    - max_hour: int: 
        The number of hours to run to. This is
        in hours.
    
    - minutes_per_hour: int: 
        The number of minutes in each hour. This
        is in minutes.
    
    - minute_interval: int: 
        The interval for a new element. This
        is in minutes.
    
    
    
    Returns
    --------
    
    - out: np.ndarray: 
        Array of time range, returned as strings.
    
    
    '''

    times = np.arange(0, max_hour*minutes_per_hour, minute_interval)

    out = np.asarray(
        [
            f"{str(h).zfill(2)}:{str(m).zfill(2)}" 
            for h, m in zip(times//minutes_per_hour, times%minutes_per_hour)
            ], 
        dtype=object,
        )
    
    return out