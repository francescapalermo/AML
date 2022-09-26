import numpy as np


def time_interval(hour:int, minute:int, interval:int)->list:
    '''
    Create list of time range that fills a day. 
    
    Edited from https://stackoverflow.com/a/58052181/19451559.
    
    Examples
    ---------

    .. code-block::

        >>> time_interval(24, 60, 30)
        ['00:00',
        '00:30',
        ...,
        '22:30',
        '23:00',
        '23:30']

    
    Arguments
    ---------
    
    - hour: int: 
        The number of hours to run to.
    
    - minute: int: 
        The number of minutes in each hour.
    
    - interval: int: 
        The interval for a new element.
    
    
    
    Returns
    --------
    
    - out: list` : 
        List of time range
    
    
    '''
    return [
        f"{str(i).zfill(2)}:{str(j).zfill(2)}" 
        for i in range(hour)
        for j in range(minute)
        if j % interval == 0
    ]