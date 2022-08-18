from tkinter import Y
import torch
import torch.nn as nn
import typing
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
import copy
import gc
 
from .utils import MyData, get_optimizer_from_name, get_criterion_from_name
from .optimizer import CombineOptimizers
from .fitting import BasicModelFitter
from .testing import BasicModelTesting
from .progress import MyProgressBar








# functions used in pytorch only training and pytorch lightning training
class TrainingHelper:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _prepare_fit_data(self, 
                            X:np.array=None, 
                            y:np.array=None,
                            train_loader:torch.utils.data.DataLoader=None,
                            X_val:typing.Union[np.array, None]=None,
                            y_val:typing.Union[np.array, None]=None,
                            val_loader:torch.utils.data.DataLoader=None,
                            ):
        '''
        This is used to prepare the fit model. Please either use 
        the `train_loader` or `X` and `y`.
        This corresponds to using either a torch DataLoader
        or a numpy array as the training data.

        Arguments
        ---------

        - `X`: `numpy.array` or `None`, optional:
            The input array to fit the model on.
            Defaults to `None`.

        - `y`: `numpy.array` or `None`, optional:
            The target array to fit the model on.
            Defaults to `None`.

        - `train_loader`: `torch.utils.data.DataLoader` or `None`, optional:
            The training data, which contains the input and the targets.
            Defaults to `None`.

        - `X_val`: `numpy.array` or `None`, optional:
            The validation input to calculate validation 
            loss on when training the model.
            Defaults to `None`

        - `X_val`: `numpy.array` or `None`, optional:
            The validation target to calculate validation 
            loss on when training the model.
            Defaults to `None`

        - `val_loader`: `torch.utils.data.DataLoader` or `None`, optional:
            The validation data, which contains the input and the targets.
            Defaults to `None`.

        Returns
        --------

        - `self`:
            The `self` that is passed as an argument.

        - `train_loader`: `torch.utils.data.DataLoader` : 
            The train data loader.
        
        - `val_loader`: `torch.utils.data.DataLoader` : 
            The validation data loader.

        '''
        assert ((train_loader is None) ^ (X is None)), 'Please either use train_loader OR X and y'
        assert ~((not val_loader is None) and (not X_val is None)), 'Please either use val_loader OR X_val and y_val'

        val_too = False if ((X_val is None) and (val_loader is None)) else True

        device_now = ('cuda' if torch.cuda.is_available() else 'cpu') if self.device == 'auto' else self.device
        self.to(device_now)

        if train_loader is None:
            if type(X) == np.ndarray:
                X = torch.from_numpy(X)
            if type(y) == np.ndarray:
                y = torch.from_numpy(y)
            train_dataset = MyData(X, y)
            train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=self.shuffle,
                                                        **self.dl_kwargs)
        if val_too:
            if val_loader is None:
                if type(X_val) == np.ndarray:
                    X_val = torch.from_numpy(X_val)
                if type(y_val) == np.ndarray:
                    y_val = torch.from_numpy(y_val)
                val_dataset = MyData(X_val, y_val)
                val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                            batch_size=self.batch_size, 
                                                            shuffle=False,
                                                            **self.dl_kwargs)
        else:
            val_loader = None

        return train_loader, val_loader

    def _prepare_predict_data(self,
                    X:np.array=None,
                    y:np.array=None,
                    test_loader:torch.utils.data.DataLoader=None, 
                    ):
        '''
        Method for building the prediction data.
        
        Arguments
        ---------
        
        - `X`: `numpy.array` or `None`, optional:
            The input array to test the model on.
            Defaults to `None`.

        - `y`: `numpy.array` or `None`, optional:
            The target array to test the model on. If set to `None`,
            then `targets_too` will automatically be set to `False`.
            Defaults to `None`.
        
        - `test_loader`: `torch.utils.data.DataLoader` or `None`, optional: 
            A data loader containing the test data.
            Defaults to `None`.
        
        
        Returns
        --------

        - `self`:
            The `self` that is passed as an argument.

        - `test_loader`: `torch.utils.data.DataLoader` : 
            The test data loader.
        
        
        '''

        assert ((test_loader is None) ^ (X is None)), 'Please either use train_loader OR X and y'

        self.return_numpy = False

        if test_loader is None:
            if type(X) == np.ndarray:
                X = torch.from_numpy(X)
                self.return_numpy = True

            if y is None:
                test_dataset = MyData(X)
            else:
                if type(y) == np.ndarray:
                    y = torch.from_numpy(y)
                    self.return_numpy = True
                test_dataset = MyData(X, y)
            
            test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                        batch_size=self.batch_size,
                                                            shuffle=False,
                                                        **self.dl_kwargs)
        
        return test_loader



    def _build_training_methods(self):
        # optimizer
        if type(self.passed_optimizer) == dict:
            self.optimizer = self._get_optimizer(self.passed_optimizer)
        else:
            self.optimizer = copy.deepcopy(self.passed_optimizer)
        
        # criterion
        if type(self.passed_criterion) == str:
            self.criterion = self._get_criterion(self.passed_criterion)
        else:
            self.criterion = copy.deepcopy(self.passed_criterion)
        
        self.built_training_method = True

        return


    def _get_params_from_names(self, names):
        '''
        Returns the layer parameters from the layers name.
        This is used to specify optimizer layers based on the
        name of the layers.
        
        
        
        Arguments
        ---------
        
        - `names`: `str`: 
            Layer name. If `'all'`, all layers will be returned.
        
        
        Raises
        ---------
        
            `TypeError`: If layer name is not an attribute of the model.
        
        Returns
        --------

        - `layer_params`.
        
        
        '''
        params = []
        for name in names:
            if hasattr(self, name):
                params += list(getattr(self, name).parameters())
            elif name == 'all':
                params += list(self.parameters())
            else:
                raise TypeError('There is no such parameter name: {}'.format(name))
        return params


    def _get_optimizer(self, opt_dict):
        '''
        Returns an optimizer, initiated with keywords.
        
        
        
        Arguments
        ---------
        
        - `opt_dict`: `dict`: 
            The optimizer name as keys, and dictionaries of 
            keywords as values. An example is:
            `
            {'adam_lap': {
                            'params':['all'], 
                            'lr':0.01, 
                            'lap_n': 20,
                            'depression_function': 'min_max_mean',
                            'depression_function_kwargs': {}
                            },
            }
            `
            The values may also be a list of optimizer keywords
            that will be used as different parameter groups in the
            optimizer. The key can also be a `torch.optim` class,
            but not initiated.
        
        Raises
        ---------
        
        - `NotImplementedError`: 
            If the values are not dictionaries or a list.
        
        Returns
        --------

        - Single `optimizer` or list of `optimizer`s,
        depending on the number of optimizers given in the 
        `opt_dict`.
        
        
        '''
        optimizer_list = []
        for optimizer_name, optimizer_kwargs in opt_dict.items():
            if type(optimizer_name) == str:
                optimizer_class = get_optimizer_from_name(optimizer_name)
            else:
                optimizer_class = optimizer_name
            if type(optimizer_kwargs) == dict:
                if 'params' in optimizer_kwargs:
                    params = self._get_params_from_names(optimizer_kwargs['params'])
                else:
                    params = self._get_params_from_names(['all'])
                optimizer = optimizer_class(params=params, 
                                            **{k:v for k,v in opt_dict[optimizer_name].items() if k != 'params'})
            elif type(optimizer_kwargs) == list:
                param_groups = []
                params = []
                for group in optimizer_kwargs:
                    if 'params' in group:
                        group['params'] = self._get_params_from_names(group['params'])
                    else:
                        group['params'] = self._get_params_from_names(['all'])
                    param_groups.append(group)
                optimizer = optimizer_class(param_groups)
            else:
                raise NotImplementedError('Either pass a dictionary or list to the optimizer keywords')
            
            optimizer_list.append(optimizer)

        if len(optimizer_list) == 1:
            return optimizer_list[0]
        else:
            return CombineOptimizers(*optimizer_list)


    def _get_criterion(self, name):
        '''
        Function that allows you to get the loss function
        by name.       
        
        
        Arguments
        ---------
        
        - `name`: `str`: 
            The name of the loss function.
        

        Returns
        --------

        - `criterion`.
        
        '''
        return get_criterion_from_name(name)




    def _resolution_calc(dim_in:int, kernel_size:int=3, 
                            stride:int=1, padding:int=0, dilation:int=1):
        '''
        Allows the calculation of resolutions after a convolutional layer.
        
        
        
        Arguments
        ---------
        
        - `dim_in`: `int`: 
            The dimension of an image before convolution is applied.
            If dim_in is a `list` or `tuple`, then two dimensions
            will be returned.
        
        - `kernel_size`: `int`, optional:
            Defaults to `3`.
        
        - `stride`: `int`, optional:
            Defaults to `1`.
        
        - `padding`: `int`, optional:
            Defaults to `0`.
        
        - `dilation`: `int`, optional:
            Defaults to `1`.
        
        
        Returns
        --------
        
        - `dim_out`: `int` : 
            The dimension size after the convolutional layer.
        
        
        '''
        if padding == 'valid':
            padding=0

        if type(dim_in) == list or type(dim_in) == tuple:
            out_h = dim_in[0]
            out_w = dim_in[1]
            out_h = (out_h + 2*padding - dilation * (kernel_size-1) - 1)/stride + 1
            out_w = (out_w + 2*padding - dilation * (kernel_size-1) - 1)/stride + 1

            return (out_h, out_w)
        
        return int(np.floor((dim_in + 2*padding -  (kernel_size - 1) - 1)/stride + 1))



    def _get_conv_params(layer):
        '''
        Given a pytorch Conv2d layer, this function can
        return a dictionary of the kernel size, stride
        and padding.
        
        
        
        Arguments
        ---------
        
        - `layer`: `torch.nn.Conv2d`: 
            Pytorch convolutional layer.       
        
        
        Returns
        --------
        
        - `params`: `dict` : 
            Dictionary containing the parameters of 
            the convolutional layer.
        
        
        '''
        kernel_size = layer.kernel_size[0] if type(layer.kernel_size) == tuple else layer.kernel_size
        stride = layer.stride[0] if type(layer.stride) == tuple else layer.stride
        padding = layer.padding[0] if type(layer.padding) == tuple else layer.padding
        return {'kernel_size': kernel_size, 'stride': stride, 'padding': padding}



















































######################## pytorch only base model

class BaseModel(TrainingHelper, nn.Module):
    '''
    A simple Auto-Encoder model that learns embeddings.
    '''
    def __init__(self, 
                    device:str='auto', 
                    batch_size:int=10,
                    n_epochs:int=10,
                    shuffle:bool=True, 
                    dl_kwargs:dict={},
                    verbose:bool=True,
                    criterion:typing.Union[str, nn.Module]='mseloss',
                    optimizer:typing.Union[typing.Dict[
                                                typing.Union[str, torch.optim.Optimizer],
                                                typing.Dict[str, typing.Any]], 
                                            torch.optim.Optimizer]={'adam': {'lr': 0.01}},
                    result_path:typing.Union[str,None]=None,
                    model_path:typing.Union[str,None]=None,
                    model_name:typing.Union[str,None]=None,
                    seed:int=None,
                    metrics_track=None,
                    ):
        '''
        An auto-encoder model, built to be run similar to sklearn models.

        Arguments
        ---------

        - `device`: `str`, optional:
            The device to use the model on. If `'auto'`,
            the model will train on CUDA if available.
            Otherwise, specify a string of the device name, for
            example `'cpu'` or `'cuda'`.
            Defaults to `'auto'`

        - `batch_size`: `int`, optional:
            The batch size to use in training and transforming.
            Only used if the input data is not a torch DataLoader.
            Defaults to `10`.

        - `n_epochs`: `int`, optional:
            The number of epochs to run the training for.
            Defaults to `10`.

        - `shuffle`: `bool`, optional:
            Whether to shuffle the training and validation data when 
            training. Only used if the input data is not a torch DataLoader.
            Defaults to `True`.

        - `dl_kwargs`: `dict`, optional:
            A dictionary of keyword arguments that
            will be passed to the training data loader.
            These will be passed to `torch.utils.data.DataLoader`.
            Only used if the input data is not a torch DataLoader.
            Defaults to `{}`.

        - `verbose`: `bool`, optional:
            Whether to print information whilst training.
            Defaults to `True`.

        - `criterion`: `str` or `torch.nn.Module`:
            The criterion that is used to calculate the loss.
            If using a string, please use one of `['mseloss', 'celoss']`
            Defaults to `mseloss`.

        - `optimizer`: `dict`, optional:
            A dictionary containing the optimizer name as keys and
            a dictionary as values containing the arguments as keys. 
            For example: `{'adam':{'lr':0.01}}`. 
            The key can also be a `torch.optim` class,
            but not initiated.
            For example: `{torch.optim.Adam:{'lr':0.01}}`. 
            Defaults to `{'adam':{'lr':0.01}}`.

        - `result_path`: `str` or `None`:
            If a string is given, a graph of the loss values will 
            be saved to this path.
            Defaults to None.

        - `model_path`: `str` or `None`:
            If a string is given, the fitted model `state_dict` will 
            be saved to this path.
            Defaults to None.

        - `model_name_suffix`: `str` or `None`:
            If a string is given, this will be added to 
            the end of the default model name, which will 
            be used when saving the model and results.
            Defaults to None.

        '''

        super(BaseModel, self).__init__()
        
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.dl_kwargs = dl_kwargs
        self.verbose = verbose
        self.passed_criterion = criterion
        self.passed_optimizer = optimizer
        self.result_path = result_path
        self.model_path = model_path
        self.metrics_track = metrics_track
        self.model_name = model_name if not model_name is None else type(self).__name__

        # training modes
        self.meta_training = False
        self.validating = False
        self.testing = False
        self.in_epoch = False
        self.in_batch = False
        self.traditional_training = False
        self.fitted=False

        if seed is None:
            seed = np.random.randint(0,1e9)
        torch.manual_seed(seed)

        return

    # the following allow for specific options during training.

    def meta_train_start(self, obj=None, **kwargs):
        self.meta_training = True
        return

    def epoch_start(self, obj=None, **kwargs):
        self.in_epoch = True
        return
    
    def batch_start(self, obj=None, **kwargs):
        self.in_batch = True
        return
    
    def batch_end(self, obj=None, **kwargs):
        self.in_batch = False
        return
    
    def val_start(self, obj=None, **kwargs):
        self.validating = True
        return
    
    def val_end(self, obj=None, **kwargs):
        self.validating = False
        return      

    def epoch_end(self, obj=None, **kwargs):
        self.in_epoch = False
        return
    
    def meta_train_end(self, obj=None, **kwargs):
        self.meta_training = False
        return

    def test_start(self, obj=None, **kwargs):
        self.testing = True
        return
    
    def traditional_train_start(self, obj=None, **kwargs):
        self.traditional_training = True
        return

    def traditional_train_end(self, obj=None, **kwargs):
        self.traditional_training = False
        return

    def test_end(self, obj=None, **kwargs):
        self.testing = False
        return

    def fit(self, 
            X:np.array=None, 
            y:np.array=None,
            train_loader:torch.utils.data.DataLoader=None,
            X_val:typing.Union[np.array, None]=None,
            y_val:typing.Union[np.array, None]=None,
            val_loader:torch.utils.data.DataLoader=None,
            train_scheduler:typing.Union[torch.optim.lr_scheduler._LRScheduler, None]=None,
            **kwargs,
            ):
        '''
        This is used to fit the model. Please either use 
        the `train_loader` or `X` and `y`.
        This corresponds to using either a torch DataLoader
        or a numpy array as the training data.

        Arguments
        ---------

        - `X`: `numpy.array` or `None`, optional:
            The input array to fit the model on.
            Defaults to `None`.

        - `y`: `numpy.array` or `None`, optional:
            The target array to fit the model on.
            Defaults to `None`.

        - `train_loader`: `torch.utils.data.DataLoader` or `None`, optional:
            The training data, which contains the input and the targets.
            Defaults to `None`.

        - `X_val`: `numpy.array` or `None`, optional:
            The validation input to calculate validation 
            loss on when training the model.
            Defaults to `None`

        - `X_val`: `numpy.array` or `None`, optional:
            The validation target to calculate validation 
            loss on when training the model.
            Defaults to `None`

        - `val_loader`: `torch.utils.data.DataLoader` or `None`, optional:
            The validation data, which contains the input and the targets.
            Defaults to `None`.

        - `train_scheduler`: `torch.optim.lr_scheduler._LRScheduler` or None, optional:
            A learning rate scheduler, used in training.
            Defaults to `None`.

        '''

        
        train_loader, val_loader = self._prepare_fit_data( 
                                                            X=X, 
                                                            y=y,
                                                            train_loader=train_loader,
                                                            X_val=X_val,
                                                            y_val=y_val,
                                                            val_loader=val_loader,
                                                            )

        self._build_model()
        self._build_training_methods()

        writer = SummaryWriter(comment='-'+self.model_name)

        self.fitting_class = BasicModelFitter(
                                                model=self, 
                                                device=self.device, 
                                                verbose=self.verbose, 
                                                model_name=self.model_name,
                                                result_path=self.result_path,
                                                model_path=self.model_path,
                                                metrics_track=self.metrics_track,
                                                writer=writer,
                                                )

        self.training_metrics = self.fitting_class.fit(
                                                        train_loader=train_loader,
                                                        n_epochs=self.n_epochs,
                                                        criterion=self.criterion,
                                                        optimizer=self.optimizer,
                                                        val_loader=val_loader,
                                                        train_scheduler=train_scheduler,
                                                        **kwargs,
                                                        )

        self.to('cpu')
        self.fitted = True
        return self.training_metrics


    @torch.no_grad()
    def predict(self,
                X:np.array=None,
                y:np.array=None,
                test_loader:torch.utils.data.DataLoader=None, 
                ):
        '''
        Method for making predictions on a test loader.
        
        Arguments
        ---------
        
        - `X_test`: `numpy.array` or `None`, optional:
            The input array to test the model on.
            Defaults to `None`.

        - `y_test`: `numpy.array` or `None`, optional:
            The target array to test the model on. If set to `None`,
            then `targets_too` will automatically be set to `False`.
            Defaults to `None`.
        
        - `test_loader`: `torch.utils.data.DataLoader` or `None`, optional: 
            A data loader containing the test data.
            Defaults to `None`.
        
        Returns
        --------
        
        - `output`: `torch.tensor` : 
            The resutls from the predictions
        
        
        '''

        test_loader = self._prepare_predict_data(
                                                X=X,
                                                y=y,
                                                test_loader=test_loader,
                                                )

        self.mt = BasicModelTesting(
                                model=self,
                                device=self.device,
                                verbose=self.verbose,
                                )
        
        output = self.mt.predict(test_loader=test_loader)

        if self.return_numpy:
            output = output.numpy()

        return output





































######################## pytorch lightning base model

class BaseLightningModule(TrainingHelper, pl.LightningModule):
    def __init__(self,
                    optimizer:typing.Union[typing.Dict[
                                                typing.Union[str, torch.optim.Optimizer],
                                                typing.Dict[str, typing.Any]], 
                                            torch.optim.Optimizer]={'adam': {'lr': 0.01}},
                    criterion:typing.Union[str, nn.Module]='mseloss',
                    n_epochs:int=10,
                    batch_size:int=10,
                    shuffle:bool=True, 
                    verbose:bool=True,
                    dl_kwargs:dict={},
                    accelerator:str='auto',
                    enable_model_summary:bool=False,
                    enable_checkpointing:bool=False,
                    pl_trainer_kwargs:dict={},
                    callbacks:list=[],
                    logging:bool=False,
                    log_every_n_steps:int=20,
                    ):
        '''
        An auto-encoder model, built to be run similar to sklearn models.
        This is built on top of `pytorch_lightning.LightningModule`.
        When training, a folder containing the pytorch and CSV logs will be
        made.


        Arguments
        ---------

        - `batch_size`: `int`, optional:
            The batch size to use in training and transforming.
            Only used if the input data is not a torch DataLoader.
            Defaults to `10`.

        - `shuffle`: `bool`, optional:
            Whether to shuffle the training data when 
            training. Only used if the input data is not a torch DataLoader.
            Defaults to `True`.

        - `dl_kwargs`: `dict`, optional:
            A dictionary of keyword arguments that
            will be passed to the training, validation and testing data loader.
            These will be passed to `torch.utils.data.DataLoader`.
            Only used if the input data is not a torch DataLoader.
            Defaults to `{}`.

        - `verbose`: `bool`, optional:
            Whether to print information whilst training.
            Defaults to `True`.

        - `criterion`: `str` or `torch.nn.Module`:
            The criterion that is used to calculate the loss.
            If using a string, please use one of `['mseloss', 'celoss']`
            Defaults to `mseloss`.

        - `optimizer`: `dict`, optional:
            A dictionary containing the optimizer name as keys and
            a dictionary as values containing the arguments as keys. 
            For example: `{'adam':{'lr':0.01}}`. 
            The key can also be a `torch.optim` class,
            but not initiated.
            For example: `{torch.optim.Adam:{'lr':0.01}}`. 
            Defaults to `{'adam':{'lr':0.01}}`.
        
        - `n_epochs`: `int`, optional:
            The number of epochs to run the training for.
            Defaults to `10`.
        
        - `accelerator`: `str`, optional:
            The device to use for training. Please use 
            any of `(“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “auto”)`.
            Defaults to `'auto'``.
        
        - `enable_model_summary`: `bool`, optional:
            Whether to pint a model summary when training.
            Defaults to `False`.

        - `logging`: `bool`, optional:
            Whether to log the run data.
            Defaults to `False`.

        - `enable_checkpointing`: `bool`, optional:
            Whether to save the model periodically.
            Defaults to `False`.

        - `pl_trainer_kwargs`: `dict`, optional:
            These are keyword arguments that will be passed to 
            `pytorch_lightning.Trainer`.
            Defaults to `{}`.

        - `callbacks`: `list`, optional:
            These are there callbacks passed to the 
            `pytorch_lightning.Trainer` class. Please don't
            pass a progress bar to this list, as the TQDM progress bar
            is passed to this list within this class.
            Defaults to `[]`.

        - `log_every_n_steps`: `int`, optional:
            How many steps to train before logging metrics.
            Defaults to `20`.

        '''
        super(BaseLightningModule, self).__init__()
        self.passed_optimizer = optimizer
        self.passed_criterion = criterion
        self.pl_trainer_kwargs = pl_trainer_kwargs
        self.n_epochs = n_epochs
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dl_kwargs = dl_kwargs
        self.callbacks = callbacks
        self.enable_model_summary = enable_model_summary
        self.verbose = verbose
        self.logging = logging
        self.log_every_n_steps = log_every_n_steps
        self.enable_checkpointing = enable_checkpointing

        self._reset_trainer()

        return
    
    def _reset_trainer(self):
        '''
        To reset the pytorch-lightning training.
        '''
        if self.logging:
            logger = [pl.loggers.CSVLogger(save_dir='./', name=f'{type(self).__name__}_logs'), 
                        pl.loggers.TensorBoardLogger(save_dir='./', name=f'{type(self).__name__}_logs')]
        else:
            logger = False

        #call_backs_to_pass = [pl.callbacks.TQDMProgressBar(refresh_rate=10)]
        #call_backs_to_pass = [ProgressBar()]

        if self.verbose:
            call_backs_to_pass = [MyProgressBar(refresh_rate=10)]
        else:
            call_backs_to_pass = []
        call_backs_to_pass.extend(self.callbacks)

        self.trainer = pl.Trainer(
                                max_epochs=self.n_epochs,
                                accelerator=self.accelerator,
                                enable_model_summary=self.enable_model_summary,
                                enable_progress_bar=self.verbose,
                                callbacks=call_backs_to_pass,
                                logger=logger,
                                log_every_n_steps=self.log_every_n_steps,
                                enable_checkpointing=self.enable_checkpointing,
                                **self.pl_trainer_kwargs
                                )

        return

    def configure_optimizers(self):
        '''
        This is required for pytorch lightning.
        If using overwriting with you own function,
        please use it to return a dictionary with 
        keys `'optimizer'` and `'lr_scheduler'`,
        if one is being used.
        The optimizer is also saved in this class as an 
        attribute `.optimizer`, which is built from
        the input, saved in `.passed_optimizer`.
        '''
        return {'optimizer': self.optimizer}


    def fit(self, 
            X:np.array=None, 
            y:np.array=None,
            train_loader:torch.utils.data.DataLoader=None,
            X_val:typing.Union[np.array, None]=None,
            y_val:typing.Union[np.array, None]=None,
            val_loader:torch.utils.data.DataLoader=None,
            ckpt_path=None,
            **kwargs,
            ):
        '''
        This is used to fit the model. Please either use 
        the `train_loader` or `X` and `y`.
        This corresponds to using either a torch DataLoader
        or a numpy array as the training data.

        Arguments
        ---------

        - `X`: `numpy.array` or `None`, optional:
            The input array to fit the model on.
            Defaults to `None`.

        - `y`: `numpy.array` or `None`, optional:
            The target array to fit the model on.
            Defaults to `None`.

        - `train_loader`: `torch.utils.data.DataLoader` or `None`, optional:
            The training data, which contains the input and the targets.
            Defaults to `None`.

        - `X_val`: `numpy.array` or `None`, optional:
            The validation input to calculate validation 
            loss on when training the model.
            Defaults to `None`

        - `X_val`: `numpy.array` or `None`, optional:
            The validation target to calculate validation 
            loss on when training the model.
            Defaults to `None`

        - `val_loader`: `torch.utils.data.DataLoader` or `None`, optional:
            The validation data, which contains the input and the targets.
            Defaults to `None`.

        '''

        self._reset_trainer()

        train_loader, val_loader = self._prepare_fit_data( 
                                                        X=X, 
                                                        y=y,
                                                        train_loader=train_loader,
                                                        X_val=X_val,
                                                        y_val=y_val,
                                                        val_loader=val_loader,
                                                        )

        self._build_training_methods()
        
        self.trainer.fit(self, 
                            train_dataloaders=train_loader, 
                            val_dataloaders=val_loader,
                            ckpt_path=ckpt_path,
                            )
        
        del train_loader
        del val_loader
        del self.trainer

        self.cpu()
        gc.collect()
        torch.cuda.empty_cache()



    def predict(self,
                X:np.array=None,
                y:np.array=None,
                test_loader:torch.utils.data.DataLoader=None, 
                ):
        '''
        Method for making predictions on a test loader.
        
        Arguments
        ---------
        
        - `X`: `numpy.array` or `None`, optional:
            The input array to test the model on.
            Defaults to `None`.

        - `y`: `numpy.array` or `None`, optional:
            The target array to test the model on. If set to `None`,
            then `targets_too` will automatically be set to `False`.
            Defaults to `None`.
        
        - `test_loader`: `torch.utils.data.DataLoader` or `None`, optional: 
            A data loader containing the test data.
            Defaults to `None`.
        
        
        Returns
        --------
        
        - `output`: `torch.tensor` : 
            The resutls from the predictions
        
        
        '''

        return_concat = test_loader is None

        self._reset_trainer()
        
        test_loader = self._prepare_predict_data( 
                                                X=X,
                                                y=y,
                                                test_loader=test_loader,
                                                )

        output = self.trainer.predict(self, dataloaders=test_loader, return_predictions=True)

        if return_concat:
            output = torch.cat(output)
            if self.return_numpy:
                output = output.numpy()

        del test_loader
        del self.trainer

        self.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        return output
