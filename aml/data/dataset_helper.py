import typing
import numpy as np
import torch
import tqdm
import joblib
from ..progress import tqdm_style
from ..parallel import ProgressParallel


class HelperDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        """
        This dataset helps you to build wrappers for
        other datasets by ensuring that any method or attribute
        of the original dataset is available as a method
        or attribute of the new dataset.

        The original dataset is available as the attribute
        :code:`._dataset`.

        Arguments
        ---------

        - dataset: torch.utils.data.Dataset:
            The dataset that will be wrapped.

        """

        self._dataset = dataset
        return

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


class WrapperDataset(HelperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        functions_index: typing.Union[typing.List[int], int, None] = None,
        functions: typing.Union[
            typing.Callable, typing.List[typing.Callable]
        ] = lambda x: x,
    ):
        """
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
            ...     dataset
            ...     functions_index=0,
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the returned
        values in the dataset by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=None,
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the first returned
        values in the dataset by 2, and the second by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=[0, 1],
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the first returned
        values in the dataset by 2, and the second by 3.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=[0, 1],
            ...     functions=[lambda x: x*2, lambda x: x*3]
            ...     )
        
        
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

            - If :code:`'all'`, then the :code:`functions` must be callable, and \
            will be applied to the output of the dataset. This allows you \
            to build a function that can act over all of the outputs of dataset. \
            The returned value will be the data that is returned by the dataset.
            
            Defaults to :code:`None`.
        
        - functions: _type_, optional:
            This is the function, or list of functions to apply to the
            corresponding indices in :code:`functions_index`. Please
            see the documentation for the :code:`functions_index` argument
            to understand the behaviour of different input types. 
            Defaults to :code:`lambda x:x`.
        
        
        """

        super(WrapperDataset, self).__init__(dataset=dataset)

        self.apply_all = False
        if functions_index is None:
            if type(functions) == list:
                self.functions = {fi: f for fi, f in enumerate(functions)}
            elif callable(functions):
                self.functions = functions
            else:
                raise TypeError(
                    "If functions_index=None, please ensure "
                    "that functions is a list or a callable object."
                )

        elif type(functions_index) == list:
            if type(functions) == list:
                assert len(functions_index) == len(
                    functions
                ), "Please ensure that the functions_index is the same length as functions."
                self.functions = {fi: f for fi, f in zip(functions_index, functions)}
            elif callable(functions):
                self.functions = {fi: functions for fi in functions_index}
            else:
                raise TypeError(
                    "If type(functions_index)==list, please ensure "
                    "that functions is a list of the same length or a callable object."
                )

        elif type(functions_index) == int:
            if callable(functions):
                self.functions = {functions_index: functions}
            else:
                raise TypeError(
                    "If type(functions_index)==int, please ensure "
                    "the functions is a callable object."
                )

        elif type(functions_index) == str:
            if functions_index == "all":
                if callable(functions):
                    self.functions = functions
                    self.apply_all = True
                else:
                    raise TypeError(
                        "Please ensure that functions is callable if functions_index == 'all'."
                    )
            else:
                raise TypeError(
                    f"{functions_index} is an invalid option for functions_index."
                )

        else:
            raise TypeError(
                "Please ensure that functions_index is a list, int or None."
            )

        return

    def __getitem__(self, index):
        if type(self.functions) == dict:
            return [
                self.functions.get(nout, lambda x: x)(out)
                for nout, out in enumerate(self._dataset[index])
            ]
        elif callable(self.functions):
            if self.apply_all:
                return self.functions(*self._dataset[index])
            else:
                return [self.functions(out) for out in self._dataset[index]]
        else:
            raise TypeError("The functions could not be applied.")

    def __len__(self):
        return len(self._dataset)


class MemoryDataset(HelperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        now: bool = True,
        verbose: bool = True,
        n_jobs: int = 1,
    ):
        """
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


        """

        super(MemoryDataset, self).__init__(dataset=dataset)

        self._data_dict = {}
        if now:

            pbar = tqdm.tqdm(
                total=len(self._dataset),
                desc="Loading into memory",
                disable=not verbose,
                smoothing=0,
                **tqdm_style,
            )

            def add_to_dict(index):
                for ni, i in enumerate(index):
                    self._data_dict[i] = self._dataset[i]
                    pbar.update(1)
                    pbar.refresh()
                return None

            all_index = np.arange(len(self._dataset))
            index_list = [all_index[i::n_jobs] for i in range(n_jobs)]

            joblib.Parallel(
                n_jobs=n_jobs,
                backend="threading",
            )(joblib.delayed(add_to_dict)(index) for index in index_list)

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
        """
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

        """
        self.inputs = inputs

    def __getitem__(self, index):
        if len(self.inputs) == 1:
            return self.inputs[0][index]
        return [x[index] for x in self.inputs]

    def __len__(self):
        return len(self.inputs[0])


class ECGCorruptor(HelperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        corrupt_sources: typing.Union[list, int, None] = None,
        noise_level: typing.Union[list, float, None] = None,
        seed: typing.Union[int, None] = None,
        axis: str = "both",
        x_noise_std: float = 0.1,
    ):
        """
        ECG Data corruptor. You may pass a noise level, sources to corrupt,
        and the seed for determining the random events. This
        class allows you to corrupt either the :code:`'x'`, :code:`'y'`, 
        or :code:`'both'`. This class is built specifically for use with
        PTB_XL (found in :code:`aml.data.datasets`).

        This function will work as expected on all devices.
        
        
        Examples
        ---------
        
        .. code-block::
        
            >>> dataset = ECGCorruptor(
            ...     dataset=dataset_train
            ...     corrupt_sources=[0,1,2,3], 
            ...     noise_level=0.5, 
            ...     )

        
        
        Arguments
        ---------
        
        - dataset: torch.utils.data.Dataset:
            The dataset to corrupt. When iterated over,
            the dataset should return :code:`x`, :code:`y`, 
            and :code:`source`.

        - corrupt_sources: typing.Union[list, int, None], optional:
            The sources to corrupt in the dataset. This can be a 
            list of sources, an integer of the source, or :code:`None`
            for no sources to be corrupted.
            Defaults to :code:`None`.

        - noise_level: typing.Union[list, int, None], optional:
            This is the level of noise to apply to the dataset. 
            It can be a list of noise levels, a single noise level to
            use for all sources, or :code:`None` for no noise.
            Defaults to :code:`None`.

        - seed: typing.Union[int, None], optional:
            This is the seed that is used to determine random events.
            Defaults to :code:`None`.

        - axis: str, optional:
            This is the axis to apply the corruption to. This
            should be either :code:`'x'`, :code:`'y'`, 
            or :code:`'both'`.
            
            - :code:`'x'`: \
            Adds a Gaussian distribution to the \
            :code:`'x'` values with :code:`mean=0` and :code:`std=0.1`.
            
            - :code:`'y'`: \
            Swaps the binary label using the function :code:`1-y_true`.
            
            - :code:`'both'`: \
            Adds a Gaussian distribution to the \
            :code:`'x'` values with :code:`mean=0` and :code:`std=0.1` \
            and swaps the binary label using the function :code:`1-y_true`.

            Defaults to :code:`'both'`.

        - x_noise_std: float, optional:
            This is the standard deviation of the noise that 
            is added to :code:`x` when it is corrupted.
            Defaults to :code:`0.1`.
        
        
        """

        assert axis in [
            "x",
            "y",
            "both",
        ], "Please ensure that the axis is from ['x', 'y', 'both']"

        super(ECGCorruptor, self).__init__()

        self._axis = axis
        self._x_noise_std = x_noise_std

        # setting the list of corrupt sources
        if corrupt_sources is None:
            self._corrupt_sources = []
        elif type(corrupt_sources) == int:
            self._corrupt_sources = [corrupt_sources]
        elif hasattr(corrupt_sources, "__iter__"):
            self._corrupt_sources = corrupt_sources
        else:
            raise TypeError(
                "Please ensure that corrupt_sources is an integer, iterable or None."
            )

        # setting the noise level
        if noise_level is None:
            self._noise_level = [0] * len(self._corrupt_sources)
        elif type(noise_level) == float:
            self._noise_level = [noise_level] * len(self._corrupt_sources)
        elif hasattr(noise_level, "__iter__"):
            if hasattr(noise_level, "__len__"):
                if hasattr(self._corrupt_sources, "__len__"):
                    assert len(noise_level) == len(self._corrupt_sources), (
                        "Please ensure that the noise level "
                        "is the same length as the corrupt sources."
                    )
            self._noise_level = noise_level
        else:
            raise TypeError(
                "Please ensure that the noise level is a float, iterable or None"
            )
        self._noise_level = {
            cs: nl for cs, nl in zip(self._corrupt_sources, self._noise_level)
        }

        if seed is None:
            rng = np.random.default_rng(None)
            seed = rng.integers(low=1, high=1e9, size=1)[0]
        self.rng = np.random.default_rng(seed)

        self._corrupt_datapoints = {"x": {}, "y": {}}

        return

    def _corrupt_x(self, index, x, y, s):
        if index in self._corrupt_datapoints["x"]:
            x = self._corrupt_datapoints["x"][index]
        else:
            g_seed_mask, g_seed_values, class_seed = self.rng.integers(
                low=1, high=1e9, size=3
            )
            self.rng = np.random.default_rng(class_seed)
            g_values = torch.Generator(device=y.device).manual_seed(int(g_seed_values))
            g_mask = torch.Generator(device=y.device).manual_seed(int(g_seed_mask))
            mask = int(
                torch.rand(size=(), generator=g_mask, device=x.device)
                > 1 - self._noise_level[s]
            )
            values = torch.normal(
                mean=0,
                std=self._x_noise_std,
                generator=g_values,
                size=x.size(),
                device=x.device,
            )
            x = x + mask * values
            self._corrupt_datapoints["x"][index] = x
        return x, y, s

    def _corrupt_y(self, index, x, y, s):
        if index in self._corrupt_datapoints["y"]:
            y = self._corrupt_datapoints["y"][index]
        else:
            g_seed_mask, class_seed = self.rng.integers(low=1, high=1e9, size=2)
            self.rng = np.random.default_rng(class_seed)
            g_mask = torch.Generator().manual_seed(int(g_seed_mask))
            if torch.rand(size=(), generator=g_mask) > 1 - self._noise_level[s]:
                y = torch.tensor(1, dtype=y.dtype, device=y.device) - y

            self._corrupt_datapoints["y"][index] = y

        return x, y, s

    @property
    def corrupt_sources(self):
        return self._corrupt_sources

    def __getitem__(self, index):
        x, y, s = self._dataset[index]
        if s in self._noise_level:
            if self._axis == "x" or self._axis == "both":
                x, y, s = self._corrupt_x(index, x, y, s)
            if self._axis == "y" or self._axis == "both":
                x, y, s = self._corrupt_y(index, x, y, s)
        return x, y, s

    def __len__(
        self,
    ):
        return len(self._dataset)
