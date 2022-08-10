import numpy as np
import typing

def make_input_roll(data:np.ndarray, sequence_length:int)->np.ndarray:
    '''
    This function will produce an array that is a rolled version of the original data sequence.
    
    Examples
    ---------
    
    ```
    >>> make_input_roll([1,2,3,4,5], sequence_length=2)
    [[1,2],[2,3],[3,4],[4,5]]
    ```


    Arguments
    ---------
    
    - `data`: `numpy.ndarray`:
        This is the data that you want transformed. Please use the shape (n_datapoints, n_features).
    
    - `sequence_length`: `int`:
        This is an integer that contains the length of each of the returned sequences.
            
    Returns:
    ---------

    - `output`: `numpy.ndarray`:
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