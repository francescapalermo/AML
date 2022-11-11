import torch
import numpy as np
import typing
import imblearn

class GroupBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self, 
        group, 
        seed:typing.Union[None, int]=None, 
        batch_size:int=20, 
        upsample:typing.Union[bool, typing.Dict[typing.Any, int]]=False,
        ):
        '''
        A pytorch batch sampler that returns a batch of samples with 
        that same group.

        Examples
        ---------

        The following will batch the training dataset
        into batches that contains single group, given 
        by the :code:`group` argument

        .. code-block::

            >>> dl = torch.utils.data.DataLoader(
            ...     train_dataset, 
            ...     batch_sampler=GroupBatchSampler(
            ...         group=train_group,
            ...         seed=seed,
            ...         batch_size=64,
            ...         )
            ...     )
        
        
        Arguments
        ---------
        
        - group:  
            The group of the data points. This should be
            the same length as the data set that is to be
            sampled.
        
        - seed: int (optional):
            Random seed for group order shuffling and 
            shuffling of points in each batch.
            Defaults to :code:`None`.
        
        - batch_size: int, (optional):
            The size of each batch. Each batch
            will be smaller than or equal in 
            size to this value.
            Defaults to :code:`20`.
        
        - upsample: typing.Union[bool, typing.Dict[typing.Any, int]], (optional):
            Whether to upsample the smaller groups,
            so that all groups have the same size.
            Defaults to :code:`False`.
        
        
        '''

        rng = np.random.default_rng(seed)
        
        group = np.asarray(group)

        upsample_bool = upsample if type(upsample) == bool else True

        if upsample_bool:
            upsample_idx, \
                group = imblearn.over_sampling.RandomOverSampler(
                    sampling_strategy='all' if type(upsample) == bool else upsample,
                    random_state=rng.integers(1e9),
                    ).fit_resample(
                        np.arange(len(group)).reshape(-1,1), 
                        group
                        )
            upsample_idx = upsample_idx.reshape(-1)

        group_unique, group_counts = np.unique(group, return_counts=True)
        group_batches = (
            np.repeat(
                np.ceil(
                    np.max(group_counts)/batch_size
                    ).astype(int), 
                len(group_unique)) 
            if upsample 
            else np.ceil(group_counts/batch_size).astype(int)
            )
        rng = np.random.default_rng(
            rng.integers(low=0, high=1e9, size=(4,))
            )
        n_batches = np.sum(group_batches)
        self.out = -1*np.ones((n_batches, batch_size))
        group_order = rng.permutation(np.repeat(group_unique, group_batches))


        for g in group_unique:
            # get the index of the items from that group
            group_idx = np.argwhere(group == g).reshape(-1)
            # shuffle the group index
            rng.shuffle(group_idx)
            # get the section of the output that we will edit
            out_temp = self.out[group_order==g].reshape(-1)
            # replace the values with the index of the items
            out_temp[:len(group_idx)] = (
                upsample_idx[group_idx] if upsample 
                else group_idx
                )
            out_temp = out_temp.reshape(-1, batch_size)
            rng.shuffle(out_temp, axis=0)
            self.out[group_order==g] = out_temp
            rng = np.random.default_rng(
                rng.integers(low=0, high=1e9, size=(3,))
                )
        
        self.out = [list(batch[batch != -1].astype(int)) for batch in self.out]

        return 

    def __iter__(self):
        return iter(self.out)
    
    def __len__(self):
        return len(self.out)
