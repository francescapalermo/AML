from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import typing
import joblib
import tqdm
import functools
import pandas as pd

from ..utils import dirtree
from ..parallel import ProgressParallel
from ..progress import tqdm_style

class TensorboardLoad:
    def __init__(
        self, 
        path:str, 
        level:typing.Union[int, None]=None,
        verbose:bool=True,
        n_jobs=1,
        ):
        '''
        This class allows you to load tensorboard files
        from a directory. 
        
        
        Arguments
        ---------
        
        - path: str: 
            The path of the directory containing the files.
        
        - level: typing.Union[int, None], optional:
            The maximum number of levels to dive into
            when loading the files. If :code:`None` then
            all levels are loaded. 
            Defaults to :code:`None`.
        
        - verbose: bool, optional:
            Whether to print progress when
            loading the files. 
            Defaults to :code:`True`.
        
        - n_jobs: int, optional:
            The number of parallel operations when loading 
            the data.
            Defaults to :code:`1`.
        
        '''

        self.level= level if not level is None else -1
        self.path = path
        self.verbose=verbose
        self.n_jobs = n_jobs

        self.file_directory = dirtree(path=path, level=level, files_only=True, file_path=True,)

        queue = [[self.file_directory, -1]]
        self.level_dict = {}
        while len(queue) > 0:
            next_item, next_item_level= queue.pop()
            if not next_item_level in self.level_dict:
                self.level_dict[next_item_level] = []

            if type(next_item) == dict:
                for key in next_item.keys():
                    if next_item[key] is None:
                        self.level_dict[next_item_level].append(key)
                    else:
                        queue.append([next_item[key], next_item_level+1])
            else:
                self.level_dict[next_item_level].extend(next_item)
        
        self.level_dict.pop(-1)

        return

    type_func_dict = {
            'scalars': 'Scalars',
            'graph': 'Graphs',
            'meta_graph': 'MetaGraph',
            'run_metadata': 'RunMetadata',
            'histograms': 'Histograms',
            'distributions': 'CompressedHistograms',
            'images': 'Images',
            'audio': 'Audio',
            'tensors': 'Tensors'
            }

    @staticmethod
    def _type_load_data(file, type_name, type_func, tags, query_expression):

        acc = EventAccumulator(file)
        acc.Reload()

        run_tags = acc.Tags()[type_name] if tags is None else tags

        results = []

        for t in run_tags:

            try:          
                events =  getattr(acc, type_func)(tag=t)
            except KeyError:
                return pd.DataFrame()

            rows = [{'value': e.value, 'step': e.step} for e in events]

            results_temp = {
                'run': file,
                'type': type_name,
                'tag': t, 
                'rows': rows, 
                }
            names = file.replace("\\", '__--__').replace("/",'__--__').split('__--__')
            names = {f'level_{nl}': name for nl, name in enumerate(names[:-1])}
            results_temp.update(names)
            
            results_temp = (pd.json_normalize(
                results_temp, 
                record_path='rows', 
                meta=['run', 'type', 'tag'] + list(names.keys())
                )
                [['run'] + list(names.keys()) + ['type', 'tag', 'step', 'value']]
                )
            results.append(results_temp)
            
        if len(results) > 0:
            results = pd.concat(results)
            if not query_expression is None:
                results = results.query(query_expression)
        else:
            results = pd.DataFrame()
        return results

    def _type_loader(
        self, 
        type_name:str,
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        if type(tags) == str:
            tags = [tags]
        elif type(tags) == list:
            pass
        elif tags is None:
            pass
        else:
            raise TypeError("Please ensure that tags is a str, list of str, or None.")

        results = {}

        # loading the event accumulators
        n_files = sum(map(len, self.level_dict.values()))
        tqdm_progress = tqdm.tqdm(
            total=n_files, 
            desc='Loading Files', 
            disable=not self.verbose,
            **tqdm_style,
            )

        for level, files in self.level_dict.items():

            parallel_func = functools.partial(
                self._type_load_data,
                type_name=type_name,
                type_func=self.type_func_dict[type_name],
                tags=tags,
                query_expression=query_expression,
                )

            parallel_comps = [
                joblib.delayed(parallel_func)( 
                    file=file,
                    )
                for file in files
                ]

            level_results = ProgressParallel(
                tqdm_bar=tqdm_progress, 
                n_jobs=self.n_jobs,
                )(parallel_comps)

            if len(level_results) > 0:
                level_results = pd.concat(level_results)
                results[level] = level_results.reset_index(drop=True)
            else:
                results[level] = pd.DataFrame()
        
        tqdm_progress.close()
        
        return results

    def scalars(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the scalars
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        values from all files, in which the accuracy
        is more than or equal to 0.5.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.scalars(
                        'Accuracy', 
                        query_expression="value >= 0.5",
                        )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='scalars', 
            tags=tags, 
            query_expression=query_expression,
            )

    def histograms(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the histograms
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        histograms from all files.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.histograms(
                        'Accuracy', 
                        )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='histograms', 
            tags=tags, 
            query_expression=query_expression,
            )

    def distributions(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the distributions
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        distributions from all files.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.distributions(
                        'Accuracy', 
                        )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='distributions', 
            tags=tags, 
            query_expression=query_expression,
            )

    def images(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the images
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        images from all files.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.images(
                        'Accuracy', 
                        )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='images', 
            tags=tags, 
            query_expression=query_expression,
            )

    def audio(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the audio
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        audio from all files.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.audio(
                        'Accuracy', 
                        )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='audio', 
            tags=tags, 
            query_expression=query_expression,
            )

    def tensors(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the tensors
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        tensors from all files.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.tensors(
                        'Accuracy', 
                        )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='tensors', 
            tags=tags, 
            query_expression=query_expression,
            )
