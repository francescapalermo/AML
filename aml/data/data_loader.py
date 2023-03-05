import numpy as np
import torch


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch).cpu().numpy()
        # return np.stack([x.cpu().numpy() for x in batch])
    elif isinstance(batch[0], (tuple, list)):
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
        """
        A wrapper for the torch dataloader that
        supplies numpy arrays instead of tensors.

        It takes all of the same arguments as
        :code:`torch.utils.data.DataLoader` and
        has the same default arguments:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

        The :code:`collate_fn` function is changed
        to allow for numpy arrays.

        The code was inspired by:
        https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

        """
        super(self.__class__, self).__init__(
            *args,
            collate_fn=numpy_collate,
            **kwargs,
        )
