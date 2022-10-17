import numpy as np
import typing
import copy as _copy
import pandas as pd



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


    Returns
    ---------

    - output: ndarray:
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

        >>> flat = flatten(
                [[[1, 2],
                [3, 4]],
                [[5, 6],
                [7, 8]]],
                start_dim=1, 
                end_dim=-1)
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






def stratification(
    array:np.ndarray, 
    **leq:float,
    ) -> np.ndarray:
    '''
    This function allows you to replace
    values in a numpy array less than or 
    equal to the given values with the keywords
    given in the function arguments. All
    values that are more than the largest
    argument will have a returned value of :code:`None`.
    
    
    
    Examples
    ---------
    
    Here, we stratify an array into 
    three groups:

    .. code-block::
    
        >>> stratification(
                np.array(
                    [[0.1, 0.2], 
                    [0.7, 0.9]]),
                Orange=0.8,
                Green=0.15,
                Red=1.0,
                )
        array(
            [['Green', 'Orange'],
            ['Orange', 'Red']], 
            dtype=object)
    
    Similarly, if the largest leq argument
    is smaller than the largest value in the
    array, then None will be returned in its place:

    .. code-block::
    
        >>> stratification(
                np.array(
                    [[0.1, 0.2], 
                    [0.7, 1.1]]),
                Orange=0.8,
                Green=0.15,
                Red=1.0,
                )
        array(
            [['Green', 'Orange'],
            ['Orange', None]], 
            dtype=object)

    
    Arguments
    ---------
    
    - array: np.ndarray: 
        Array to perform transformation over.
    
    - **leq: int:
        The keyword arguments used to 
        stratify the object. The keywords
        are the labels in the output array.
    
    Returns
    --------
    
    - out: np.ndarray: 
        Array, stratified into the groups
        given by the arguments. 
    
    
    '''

    # setting output array
    array_out = np.empty(array.shape, dtype=object)

    # finding the order to apply <= (LEQ)
    leq_order = [k for k, _ in sorted(leq.items(), key=lambda item: item[1], reverse=True)]

    # masking is used to avoid nan values
    for key in leq_order:
        mask = np.zeros(array.shape, dtype=bool)
        np.less_equal(array, leq[key], out=mask, where=~pd.isna(array))
        array_out[mask] = key

    # copying values where the original has NA values.
    array_out[pd.isna(array)] = array[pd.isna(array)]

    return array_out