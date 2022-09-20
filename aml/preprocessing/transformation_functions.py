import numpy as np
import typing
import copy as _copy



def make_input_roll(
    data:np.ndarray, 
    sequence_length:int,
    )->np.ndarray:
    '''
    This function will produce an array that is a rolled 
    version of the original data sequence. The original 
    sequence must be 2D.
    
    Examples
    ---------
    
    .. code-block::

        >>> make_input_roll(np.array([[1],[2],[3],[4],[5]]), sequence_length=2)
        array([[[1],
                [2]],

                [[2],
                [3]],

                [[3],
                [4]],

                [[4],
                [5]]]
    
    similarly:

    .. code-block::

        >>> make_input_roll(np.array([[1, 2],[3, 4],[5, 6]]), sequence_length=2)
        array([[[1,2],
                [3,4]],

                [[3,4],
                [5,6]]]


    Arguments
    ---------
    
    - data: numpy.ndarray:
        This is the data that you want transformed. Please use the shape (n_datapoints, n_features).
    
    - sequence_length: int:
        This is an integer that contains the length of each of the returned sequences.


    Returns:
    ---------

    - output: numpy.ndarray:
        This is an array with the rolled data.
    
    '''
    assert type(sequence_length) == int, "Please ensure that sequence_length is an integer"
    
    if data.shape[0] < sequence_length:
        raise TypeError("Please ensure that the input can be rolled "\
                        "by the specified sequence_length. Input size was "\
                        f"{data.shape} and the sequence_length was {sequence_length}.")
    
    output = data[
        np.lib.stride_tricks.sliding_window_view(
            np.arange(data.shape[0]), 
            window_shape=sequence_length
            )
        ]
  
    return output





def flatten(
    data:np.ndarray,
    start_dim:int=0,
    end_dim:int=-1,
    copy=False,
    )->np.ndarray:
    '''
    This class allows you to flatten an array inside a pipeline.
    This class was implemented to mirror the behaviour in 
    :code:`https://pytorch.org/docs/stable/generated/torch.flatten.html`.
    
    
    Examples
    ---------

    .. code-block::

        >>> flat = Flatten(start_dim=1, end_dim=-1)
        >>> flat.fit(None, None) # ignored
        >>> flat.transform(
                [[[1, 2],
                [3, 4]],
                [[5, 6],
                [7, 8]]]
                )
        [[1,2,3,4],
        [5,6,7,8]]



    Arguments
    ---------
    
    - start_dim: int, optional:
        The first dim to flatten. 
        Defaults to :code:`0`.
    
    - end_dim: int, optional:
        The last dim to flatten. 
        Defaults to :code:`-1`.
    
    - copy: bool, optional:
        Whether to return a copied version
        of the array during the transform
        method.
        Defaults to :code:`False`.
    
    '''

    if copy: out = _copy.deepcopy(data)
    else: out = data

    # the starting dims not flattened
    new_shape = [out.shape[i] for i in range(start_dim)]
    # the flattened dim
    new_shape.append(-1)
    # adding the non flattened dims to the end
    if end_dim != -1:
        new_shape.extend([out.shape[i] for i in range(end_dim+1, len(out.shape))])
    
    return out.reshape(*new_shape)





















def make_input_roll_old(data:np.ndarray, sequence_length:int)->np.ndarray:
    '''
    This function will produce an array that is a rolled 
    version of the original data sequence. The original 
    sequence must be 2D.
    
    Examples
    ---------

    .. code-block::

        >>> make_input_roll(np.array([[1],[2],[3],[4],[5]]), sequence_length=2)
        array([[[1],
                [2]],

                [[2],
                [3]],

                [[3],
                [4]],

                [[4],
                [5]]]



    Arguments
    ---------
    
    - data: numpy.ndarray:
        This is the data that you want transformed. Please use the shape (n_datapoints, n_features).
    
    - sequence_length: int:
        This is an integer that contains the length of each of the returned sequences.


    Returns:
    ---------

    - output: numpy.ndarray:
        This is an array with the rolled data.
    
    '''
    assert type(sequence_length) == int, "Please ensure that sequence_length is an integer"
    
    if data.shape[0] < sequence_length:
        raise TypeError("Please ensure that the input can be rolled "\
                        "by the specified sequence_length. Input size was "\
                        f"{data.shape} and the sequence_length was {sequence_length}.")
    
    output = np.empty((data.shape[0]-sequence_length + 1,
                sequence_length, 
                data.shape[1]), dtype = data.dtype)
    
    for ns in range(sequence_length):
        endpoint = sequence_length - ns - 1
        
        if endpoint == 0:
            for n1 in range(data.shape[1]):
                output[:,ns,n1] = data[ns:,n1]


        else:
            for n1 in range(data.shape[1]):
                output[:,ns,n1] = data[ns:-endpoint,n1]
  
    return output
