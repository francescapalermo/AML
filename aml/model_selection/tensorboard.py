from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
import typing
import joblib
import tqdm
import functools
import pandas as pd

from ..utils.directory import dirtree
from ..utils.parallel import ProgressParallel
from ..utils.progress import tqdm_style

class TBToPD:
    def __init__(
        self, 
        path:str, 
        level:typing.Union[int, None]=None,
        n_jobs:int=1,
        verbose:bool=True,
        ):


        assert n_jobs == 1, "This currently only supports 1 thread."
        self.level= level if not level is None else -1
        self.path = path
        self.n_jobs = n_jobs
        self.verbose=verbose

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

    @staticmethod
    def _scalars_run_tag(tag, run, acc, query_expression):

        try:          
            events =  acc.Scalars(run=run, tag=tag)
        except KeyError:
            return pd.DataFrame()

        rows = [{'value': e.value, 'step': e.step} for e in events]

        results = {
            'run': run, 
            'tag': tag, 
            'rows': rows, 
            }
        names = run.replace("\\", '__--__').replace("/",'__--__').split('__--__')
        names = {f'level_{nl}': name for nl, name in enumerate(names[:-1])}
        results.update(names)
        
        results = (pd.json_normalize(
            results, 
            record_path='rows', 
            meta=['run', 'tag'] + list(names.keys())
            )
            [['run'] + list(names.keys()) + ['tag', 'step', 'value']]
            )

        if not query_expression is None:
            results = results.query(query_expression)
        
        return results

    def scalars(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the scalars
        written to the tensorboard, with the give
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        values from all files, in which the accuracy
        is more than or equal to 0.5.

        .. code_block::
        
            >>> tbpd = TBToPD('./')
            >>> tbpd.scalars(
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
        if type(tags) == str:
            tags = [tags]
        elif type(tags) == list:
            pass
        elif tags is None:
            pass
        else:
            raise TypeError("Please ensure that tags is a str, list of str, or None.")

        results = {}

        runs = {}
        n_files = 0
        for level, files in tqdm.tqdm(
            self.level_dict.items(),
            desc='Loading Level',
            **tqdm_style,
            ):
            acc = EventMultiplexer()
            for file in files:
                acc.AddRun(file)
            acc.Reload()
            level_runs = acc.Runs()
            runs[level] = level_runs
            for run in level_runs:
                n_files += len(level_runs[run]['scalars'] if tags is None else tags)

        tqdm_progress = tqdm.tqdm(
            total=n_files, 
            desc='Reading Files', 
            disable=not self.verbose,
            **tqdm_style,
            )

        for level in self.level_dict.keys():
            level_runs = runs[level]
            parallel_func = functools.partial(
                self._scalars_run_tag,
                acc=acc,
                query_expression=query_expression,
                )

            parallel_comps = [
                joblib.delayed(parallel_func)(
                    tag=tag, 
                    run=run,
                    )
                for run in level_runs 
                for tag in (level_runs[run]['scalars'] if tags is None else tags)
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
        
        return results