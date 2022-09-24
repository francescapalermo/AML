import numpy as np
import importlib.util



def module_from_file(module_name:str, file_path:str):
    '''
    Will open a module from a file path.
    
    Edited from https://stackoverflow.com/a/51585877/19451559.
    
    Examples
    ---------

    .. code-block::

        >>> validated_date = module_from_file(
                "validated_date", 
                "../data/manual_uti_labels.py"
                ).validated_date
        >>> validated_date
        <function validated_date.validated_date(return_df=False)>

    
    Arguments
    ---------
    
    - module_name: str: 
        The name of the module to load.
    
    - file_path: str: 
        File path to that module.
    
    
    
    Returns
    --------
    
    - out: module` : 
        A python module that can be 
        used to access objects from 
        within it.
    
    
    '''
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



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