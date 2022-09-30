import typing
import ast
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import os
from torchvision.datasets.utils import download_and_extract_archive
import tqdm
import joblib

from ..progress.progress import tqdm_style
from ..parallel.parallel import ProgressParallel

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
        the original dataset will still be available.
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
            The number of parallel
            loads into memory.
            This is ignored if :code:`now=False`. 
            Currently :code:`n_jobs=1` runs fastest...
            Defaults to :code:`1`.
        
        
        '''

        self.dataset = dataset
        self.data_dict = {}
        if now:
            pbar = tqdm.tqdm(
                desc='Loading into memory',
                total=len(dataset),
                disable=not verbose,
                **tqdm_style
                )

            def return_value(index):
                return [index, dataset[index]]

            data_point_list = ProgressParallel(
                pbar, 
                n_jobs=n_jobs,
                )(
                    joblib.delayed(return_value)(index)
                    for index in range(len(dataset))
                    )
            
            self.data_dict = {
                data_point[0]:data_point[1]
                for data_point in data_point_list
                }

        return

    def __getitem__(self, index):

        if index in self.data_dict:
            return self.data_dict[index]
        else:
            output = self.dataset[index]
            self.data_dict[index] = output
            return output
    
    def __len__(self):
        return len(self.dataset)
    
    def __getattr__(self, name):
        return getattr(self.dataset, name)



class PTB_XL(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path:str='./',
        train:bool=True,
        sampling_rate:typing.Literal[100, 500]=100,
        source_name:typing.Literal['nurse', 'site', 'device']='nurse',
        return_sources:bool=True,
        ):
        '''
        ECG Data, as described here: https://physionet.org/content/ptb-xl/1.0.2/.
        
        
        
        Examples
        ---------
        
        .. code-block::
        
            >>> dataset = PTB_XL(
                    data_path='../../data/', 
                    train=True, 
                    source_name='nurse', 
                    sampling_rate=500,
                    return_sources=False,
                    )

        
        
        Arguments
        ---------
        
        - data_path: str, optional:
            The path that the data is saved
            or will be saved. 
            Defaults to :code:`'./'`.
        
        - train: bool, optional:
            Whether to load the training or testing set. 
            Defaults to :code:`True`.
        
        - sampling_rate: typing.Literal[100, 500], optional:
            The sampling rate. This should be
            in :code:`[100, 500]`. 
            Defaults to :code:`100`.
        
        - source_name: typing.Literal['nurse', 'site', 'device'], optional:
            Which of the three attributes should be 
            interpretted as the data sources. This should
            be in  :code:`['nurse', 'site', 'device']`.
            This is ignored if :code:`return_sources=False`.
            Defaults to :code:`'nurse'`.
        
        - return_sources: bool, optional:
            Whether to return the sources alongside
            the data and targets. For example, with 
            :code:`return_sources=True`, for every index
            this dataset will return :code:`data, target, source`. 
            Defaults to :code:`True`.
        
        
        '''
        try:
            import wfdb
        except ImportError:
            raise ImportError('Please install wfdb first.')

        assert sampling_rate in [100, 500], \
            "Please choose sampling_rate from [100, 500]"
        assert type(train) == bool, "Please use train = True or False"
        assert source_name in ['nurse', 'site', 'device'], \
            "Please choose source_name from ['nurse', 'site', 'device']"

        
        self.data_path = data_path
        self.download()
        self.data_path = os.path.join(
            self.data_path, 
            'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2/',
            )

        self.train=train
        self.sampling_rate = sampling_rate
        self.source_name = source_name
        self.return_sources = return_sources
        self.meta_data = pd.read_csv(self.data_path+'ptbxl_database.csv')
        self.meta_data['scp_codes'] = (self.meta_data
            ['scp_codes']
            .apply(lambda x: ast.literal_eval(x))
            )
        self.aggregate_diagnostic() # create diagnostic columns
        self.meta_data = self.meta_data[~self.meta_data[self.source_name].isna()]

        if self.train:
            self.meta_data = self.meta_data.query("strat_fold != 10")
        else:
            self.meta_data = self.meta_data.query("strat_fold == 10")
        
        self.targets = self.meta_data[['NORM', 'CD', 'HYP', 'MI', 'STTC']].values
        self.sources = self.meta_data[self.source_name].values

        return

    def _check_exists(self):
        folder = os.path.join(
            self.data_path, 
            'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2',
            )
        return os.path.exists(folder)
        
    def download(self):
        
        if self._check_exists():
            print("Files already downloaded.")
            return

        download_and_extract_archive(
            url='https://physionet.org/static'\
                '/published-projects/ptb-xl/'\
                'ptb-xl-a-large-publicly-available'\
                '-electrocardiography-dataset-1.0.2.zip',
            download_root=self.data_path,
            extract_root=self.data_path,
            filename='ptbxl.zip',
            remove_finished=True
            )

        return

    @staticmethod
    def single_diagnostic(y_dict, agg_df):
        tmp = []
        for key in y_dict.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def aggregate_diagnostic(self):
        agg_df = pd.read_csv(self.data_path +'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        self.meta_data['diagnostic_superclass'] = (self.meta_data
            ['scp_codes']
            .apply(
                self.single_diagnostic, 
                agg_df=agg_df,
                )
            )
        mlb = MultiLabelBinarizer()
        self.meta_data = self.meta_data.join(
            pd.DataFrame(
                mlb.fit_transform(
                    self.meta_data.pop('diagnostic_superclass')
                    ),
                columns=mlb.classes_,
                index=self.meta_data.index,
                )
            )
        return

    def __getitem__(self, index):
        data = self.meta_data.iloc[index]

        if self.sampling_rate == 100:
            f = data['filename_lr']
            x = wfdb.rdsamp(self.data_path+f)
        elif self.sampling_rate == 500:
            f = data['filename_hr']
            x = wfdb.rdsamp(self.data_path+f)
        x = torch.tensor(x[0]).transpose(0,1).float()
        y = torch.tensor(
            data
            [['NORM', 'CD', 'HYP', 'MI', 'STTC']]
            .values
            .astype(np.int64)
            ).float()
        source = data[self.source_name]

        if self.return_sources:
            return x, y, source
        else:
            return x, y
    
    def __len__(self):
        return len(self.meta_data)
    