import typing
import numpy as np
import torch
import tqdm
import joblib
from ..progress import tqdm_style
from ..parallel import ProgressParallel


class WrapperDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset:torch.utils.data.Dataset,
        functions_index:typing.Union[typing.List[int], int, None]=None,
        functions:typing.Union[typing.Callable, typing.List[typing.Callable]]=lambda x: x,
        ):
        '''
        This allows you to wrap a dataset with a set of 
        functions that will be applied to each returned 
        data point. You can apply a single function to all 
        outputs of a data point, or a different function
        to each of the different outputs.
        
        
        
        Examples
        ---------

        The following would multiply all of the first returned
        values in the dataset by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
                dataset
                functions_index=0,
                functions=lambda x: x*2
                )

        The following would multiply all of the returned
        values in the dataset by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
                dataset
                functions_index=None,
                functions=lambda x: x*2
                )

        The following would multiply all of the first returned
        values in the dataset by 2, and the second by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
                dataset
                functions_index=[0, 1],
                functions=lambda x: x*2
                )

        The following would multiply all of the first returned
        values in the dataset by 2, and the second by 3.
        
        .. code-block::
        
            >>> WrapperDataset(
                dataset
                functions_index=[0, 1],
                functions=[lambda x: x*2, lambda x: x*3]
                )
        
        
        Arguments
        ---------
        
        - dataset: torch.utils.data.Dataset: 
            The dataset to be wrapped.
        
        - functions_index: typing.Union[typing.List[int], int, None], optional:
            The index of the functions to be applied to. 

            - If :code:`None`, then if the :code:`functions` is callable, it \
            will be applied to all outputs of the data points, \
            or if the :code:`functions` is a list, it will be applied to the corresponding \
            output of the data point.

            - If :code:`list` then the corresponding index will have the \
            :code:`functions` applied to them. If :code:`functions` is a list, \
            then it will be applied to the corresponding indicies given in :code:`functions_index` \
            of the data point. If :code:`functions` is callable, it will be applied to all of the \
            indicies in :code:`functions_index`
        
            - If :code:`int`, then the :code:`functions` must be callable, and \
            will be applied to the output of this index.
            
            Defaults to :code:`None`.
        
        - functions: _type_, optional:
            This is the function, or list of functions to apply to the
            corresponding indices in :code:`functions_index`. Please
            see the documentation for the :code:`functions_index` argument
            to understand the behaviour of different input types. 
            Defaults to :code:`lambda x:x`.
        
        
        '''

        self._dataset = dataset
        if functions_index is None:
            if type(functions) == list:
                self.functions = {fi: f for fi, f in enumerate(functions)}
            elif callable(functions):
                self.functions=functions
            else:
                raise TypeError("If functions_index=None, please ensure "\
                    "that functions is a list or a callable object.")
        
        elif type(functions_index) == list:
            if type(functions) == list:
                assert len(functions_index) == len(functions), \
                    "Please ensure that the functions_index is the same length as functions."
                self.functions = {fi: f for fi, f in zip(functions_index, functions)}
            elif callable(functions):
                self.functions = {fi: functions for fi in functions_index}
            else:
                raise TypeError("If type(functions_index)==list, please ensure "\
                    "that functions is a list of the same length or a callable object.")

        elif type(functions_index) == int:
            if callable(functions):
                self.functions = {functions_index: functions}
            else:
                raise TypeError("If type(functions_index)==int, please ensure "\
                    "the functions is a callable object.")

        else:
            raise TypeError("Please ensure that functions_index is a list, int or None.")

        return

    def __getitem__(self, index):
        if type(self.functions) == dict:
            return [
                self.functions.get(nout, lambda x: x)(out) 
                for nout, out in enumerate(self._dataset[index])
                ]
        elif callable(self.functions):
            return [self.functions(out) for out in self._dataset[index]]
        else:
            raise TypeError("The functions could not be applied.")
    
    def __len__(self):
        return len(self._dataset)

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        else:
            raise AttributeError



class MemoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset:torch.utils.data.Dataset,
        now:bool=True,
        verbose:bool=True,
        n_jobs:int=1,
        ):
        '''
        This dataset allows the user
        to wrap another dataset and 
        load all of the outputs into memory,
        so that they are accessed from RAM 
        instead of storage. All attributes of
        the original dataset will still be available, except
        for :code:`._dataset` and :code:`._data_dict` if they 
        were defined.
        It also allows the data to be saved in memory right
        away or after the data is accessed for the first time.
               
        
        Examples
        ---------
        
        .. code-block::
        
            >>> dataset = MemoryDataset(dataset, now=True)
        
        
        Arguments
        ---------
        
        - dataset: torch.utils.data.Dataset: 
            The dataset to wrap and add to memory.
        
        - now: bool, optional:
            Whether to save the data to memory
            right away, or the first time the 
            data is accessed. If :code:`True`, then
            this initialisation might take some time
            as it will need to load all of the data.
            Defaults to :code:`True`.
        
        - verbose: bool, optional:
            Whether to print progress
            as the data is being loaded into
            memory. This is ignored if :code:`now=False`.
            Defaults to :code:`True`.
        
        - n_jobs: int, optional:
            The number of parallel operations when loading 
            the data to memory.
            Defaults to :code:`1`.
        
        
        '''

        self._dataset = dataset
        self._data_dict = {}
        if now:

            pbar = tqdm.tqdm(
                total = len(dataset),
                desc='Loading into memory',
                disable=not verbose,
                **tqdm_style
                )

            def add_to_dict(index):
                for ni, i in enumerate(index):
                    self._data_dict[i] = dataset[i]
                    pbar.update(1)
                    pbar.refresh()
                return None

            all_index = np.arange(len(dataset))
            index_list = [all_index[i::n_jobs] for i in range(n_jobs)]

            joblib.Parallel(
                n_jobs=n_jobs,
                backend='threading',
                )(
                    joblib.delayed(add_to_dict)(index)
                    for index in index_list
                    )
            
            pbar.close()

        return

    def __getitem__(self, index):

        if index in self._data_dict:
            return self._data_dict[index]
        else:
            output = self._dataset[index]
            self._data_dict[index] = output
            return output
    
    def __len__(self):
        return len(self._dataset)

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        else:
            raise AttributeError



class MyData(torch.utils.data.Dataset):
    def __init__(self, *inputs: torch.tensor):
        '''
        Allows the user to turn any set of tensors 
        into a dataset.
        
        Examples
        ---------
        
        .. code-block:: 
        
            >>> data = MyData(X, y, other)
            >>> len(data) == len(X)
            True
        
        
        Arguments
        ---------

        - inputs: torch.tensor:
            Any tensors.
        
        '''
        self.inputs = inputs
    def __getitem__(self,index):
        if len(self.inputs) == 1:
            return self.inputs[0][index]
        return [x[index] for x in self.inputs]
    def __len__(self):
        return len(self.inputs[0])