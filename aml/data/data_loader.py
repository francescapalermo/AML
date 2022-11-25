import numpy as np
import torch


def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        *args,
        **kwargs,
        ):
        '''
        A wrapper for the torch dataloader that 
        supplies numpy arrays instead of tensors.

        It takes all of the same arguments as 
        :code:`torch.utils.data.DataLoader` and
        has the same default arguments.

        The :code:`collate_fn` function is changed
        to allow for numpy arrays.
        
        '''
        super(self.__class__, self).__init__(
            *args, 
            collate_fn=numpy_collate,
            **kwargs,
            )