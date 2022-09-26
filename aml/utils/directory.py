import typing
import os

def dirtree(
    path:str, 
    level:typing.Union[None, int]=None, 
    files_only:bool=False,
    file_path:bool=False
    ) -> dict:
    '''
    This function will produce a dictionary of 
    the file structure. All keys with value of 
    :code:`None` are files, and if 
    :code:`files_only=True` all values that are 
    part of a list are files.
    
    
    
    Examples
    ---------
    
    .. code-block:: 
    
        >>> dirtree('./')
    
    
    Arguments
    ---------
    
    - path: str: 
        The path to search over.
    
    - level: typing.Union[None, int], optional:
        The number of levels to recursively search.
        :code:`level=0` is the files in the directory of the path,
        and :code:`level=1` would be all of the files in the directories
        of the directory of the path, etc. 
        :code:`None` searches recursively until there are no 
        more directories in the leaves.
        Defaults to :code:`None`.
    
    - files_only: bool, optional:
        Whether to only show the files, or the folders too.
        :code:`True` will only return the files.
        Defaults to :code:`False`.
    
    - file_path: bool, optional:
        If :code:`True`, then the returned
        names will contain the full paths.
        Defaults to :code:`False`.
    
    Returns
    ---------
    
    - directory_dict: dict:
        The dictionary containing the file structure.
    
    '''
    def recursive_build(path, level):
        if level is None:
            level = -1
        if os.path.isdir(path):
            if level != 0:
                d = {}
                for name in os.listdir(path):
                    if file_path:
                        d[os.path.join(path, name)] = recursive_build(os.path.join(path, name), level=level-1)
                    else:
                        d[name] = recursive_build(os.path.join(path, name), level=level-1)
            else:
                if files_only:
                    if file_path:
                        d = [os.path.join(path, name) for name in os.listdir(path) if not os.path.isdir(os.path.join(path, name))]
                    else:
                        d = [name for name in os.listdir(path) if not os.path.isdir(os.path.join(path, name))]
                else:
                    if file_path:
                        d = [os.path.join(path, name) for name in os.listdir(path)]
                    else:
                        d = [name for name in os.listdir(path)]
        else:
            d=None
        return d
    return {path: recursive_build(path, level)}