import torch
import numpy as np
import typing
import imblearn

class GroupBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self, 
        group:typing.Union[np.ndarray, typing.List[typing.Any]], 
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
        
        - group: typing.Union[np.ndarray, typing.List[typing.Any]]:
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


class GroupAllBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self, 
        group:typing.Union[np.ndarray, typing.List[typing.Any]], 
        seed:typing.Union[None, int]=None, 
        shuffle_groups:bool=True,
        shuffle_points:bool=True,
        ):
        '''
        A pytorch batch sampler that returns batches containing
        all of the data for an entire group, in a shuffled order.
        There will only be as many batches as number of unique
        groups, since each batch contains all data from that group.

        Examples
        ---------

        The following will batch the training dataset
        into batches that contains all data from a 
        single group, given by the :code:`group` argument.

        .. code-block::

            >>> dl = torch.utils.data.DataLoader(
            ...     train_dataset, 
            ...     batch_sampler=GroupAllBatchSampler(
            ...         group=train_group,
            ...         seed=seed,
            ...         )
            ...     )
        
        
        Arguments
        ---------
        
        - group: typing.Union[np.ndarray, typing.List[typing.Any]]:
            The group of the data points. This should be
            the same length as the data set that is to be
            sampled.
        
        - seed: int (optional):
            Random seed for group order shuffling and 
            shuffling of points in each batch.
            Defaults to :code:`None`.
        
        - shuffle_groups: bool (optional):
            Shuffles the order of the groups that are
            returned.
            Defaults to :code:`True`.
        
        - shuffle_points: bool (optional):
            Shuffles the order of the points that are
            returned in the groups.
            Defaults to :code:`True`.
        
        
        '''

        rng = np.random.default_rng(seed)
        rng = np.random.default_rng(
            rng.integers(low=0, high=1e9, size=(2,))
            )
        
        group = np.asarray(group)

        group_unique = np.unique(group)
        group_order = rng.permutation(group_unique) if shuffle_groups else group_unique

        self.out = []
        for g in group_order:
            # get the index of the items from that group
            group_idx = np.argwhere(group == g).reshape(-1)
            # shuffle the group index
            if shuffle_points:
                rng.shuffle(group_idx)
            self.out.append(list(group_idx.astype(int)))
        return 

    def __iter__(self):
        return iter(self.out)
    
    def __len__(self):
        return len(self.out)


class _SplitSampler(torch.utils.data.Sampler):
    def __init__(
        self, 
        sampler:torch.utils.data.Sampler, 
        split_start=None, 
        split_end=None,
        ):
        self.split_start = 0 if split_start is None else split_start
        self.split_end = 1 if split_end is None else split_end
        self.sampler = sampler

    def __iter__(self):
        for idx in self.sampler:
            start_idx = int(len(idx)*self.split_start)
            end_idx = int(len(idx)*self.split_end)
            yield idx[start_idx:end_idx]
    
    def __len__(self):
        return len(self.sampler)
    


def split_sampler(
    sampler:torch.utils.data.Sampler,
    splits:typing.Union[None, typing.List[float]]=None,
    ) -> typing.List[torch.utils.data.Sampler]:
    '''
    This allows you to split a sampler into multiple samplers
    based on proportions.
    The splits will be taken in order given and each sampler
    will contain data between each pair of proportions.

    Examples
    ---------

    The following is an example of how to 
    split a sampler into two:

    .. code-block::

        >>> sampler = [[0,1], [2,3], [4,5]]
        >>> s1, s2 = split_sampler(sampler, splits=[0.5, 0.5])
        >>> for i in zip(s1, s2):
        ...     print(i)
        ([0], [1])
        ([2], [3])
        ([4], [5])
    
    The following is what would happen if batches 
    were of odd length::

        >>> sampler = [[0,1,2], [3,4,5], [6,7,8]]
        >>> s1, s2 = split_sampler(sampler, splits=[0.5, 0.5])
        >>> for i in zip(s1, s2):
        ...     print(i)
        ([0], [1, 2])
        ([3], [4, 5])
        ([6], [7, 8])
    
    The following is how you would split a sampler 
    into three::

        >>> sampler = [[0,1,2], [3,4,5], [6,7,8]]
        >>> s1, s2, s3 = split_sampler(sampler, splits=[0.34, 0.34, 0.34])
        >>> for i in zip(s1, s2, s3):
        ...     print(i)
        ([0], [1], [2])
        ([3], [4], [5])
        ([6], [7], [8])

    And an example with this in use with 
    :code:`GroupAllBatchSampler`::

        >>> sampler = GroupAllBatchSampler(
        ...     group=np.array([1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]),
        ...     seed=42,
        ...     shuffle_groups=True,
        ...     shuffle_points=True,
        ...     )
        >>> sampler_splits = split_sampler(sampler, splits=[0.34, 0.34, 0.34])
        >>> for i in zip(*sampler_splits):
        ...     print(i)
        ([11], [9], [10])
        ([0], [1], [2])
        ([14], [13], [12])
        ([5], [4], [3])
        ([8], [6], [7])


    Arguments
    ---------

    - sampler: torch.utils.data.Sampler:
        This is the sampler to split.
    
    - splits: typing.Union[None, typing.List[float]] (optional):
        This is the proportions of each batch to be 
        contained in each new sampler.
        If :code:`None`, then the original sampler
        is returned in a list.
        Please see the examples. 
        Defaults to :code:`None`.
    

    Returns
    ---------

    - samplers: list:
        A list of the new samplers.

    '''

    samplers = []
    if splits is None:
        samplers.append(sampler)
    elif len(splits) < 2:
        raise TypeError("If a split is given, please supply at least two values. "\
            "For example, if splitting in half, use [0.5, 1.0]")
    else:
        splits = np.cumsum(splits)
        splits = [[l0, l1] for l0, l1 in zip(splits[:-1], splits[1:])]
        splits.insert(0, [0.0, splits[0][0]])
        for l0, l1 in splits:
            samplers.append(_SplitSampler(sampler, split_start=l0, split_end=l1))
    
    return samplers



class _SequentialSamplerSampler(torch.utils.data.Sampler):
    def __init__(self, *samplers:torch.utils.data.Sampler):
        '''
        This sampler allows you to combine several samplers
        sequentially, so that each one can be run after the other.

        Arguments
        ---------

        - samplers: torch.utils.data.Sampler:
            The samplers to combine.
        
        '''
        self.samplers = samplers
    
    def __iter__(self):
        for sampler in self.samplers:
            for idx in sampler:
                yield idx
    
    def __len__(self):
        length = 0
        for sampler in self.samplers:
            length += len(sampler)
        return length


def sequential_samplers_sampler(*samplers)->torch.utils.data.Sampler:
    '''
    This sampler allows you to combine several samplers
    sequentially, so that each one can be run after the other.

    Arguments
    ---------

    - samplers: torch.utils.data.Sampler:
        The samplers to combine.
    

    Returns
    ---------
    
    - sampler: torch.utils.data.Sampler:
        A single sampler that will use each 
        of the given samplers in sequence.

    '''

    return _SequentialSamplerSampler(*samplers)

