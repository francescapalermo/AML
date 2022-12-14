import importlib.util



def module_from_file(module_name:str, file_path:str):
    '''
    Will open a module from a file path.
    
    Edited from https://stackoverflow.com/a/51585877/19451559.
    
    Examples
    ---------

    .. code-block::

        >>> model_trainer = module_from_file(
        ...     'model_trainer', 
        ...     './some_code/model_trainer.py'
        ...     )
        >>> train_function = model_trainer.train
        <function model_trainer.train(...)>

    
    Arguments
    ---------
    
    - module_name: str: 
        The name of the module to load.
    
    - file_path: str: 
        File path to that module.
    
    
    
    Returns
    --------
    
    - out: module: 
        A python module that can be 
        used to access objects from 
        within it.
    
    
    '''
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


