import torch
from torch import nn
import numpy as np
import typing
from copy import deepcopy

from .base_model import BaseLightningModule
from .utils import get_function_from_name



class FCLayers(nn.Module):
    def __init__(self, 
                    n_input:int, 
                    n_output:int, 
                    hidden_layer_sizes:typing.List[int]=(100,),
                    activation:typing.Union[str, torch.nn.Module]='relu',
                    use_softmax:bool=True, 
                    dropout=0.2,
                    ):
        '''
        This class can be used on its own as an MLP.

        Arguments
        ---------
            
        - n_input: int:
            The size of the input feature dimension.

        - n_output: int:
            The output dimension sizes.

        - hidden_layer_sizes: typing.List[int], optional:
            The hidden layer sizes.
            Defaults to :code:`(100,)`.

        - activation: typing.Union[str,torch.nn.Module] , optional:
            The activation function to be used in the hidden layers to add
            non-linearity. You may pass a str of the form:
            - 'identity': The identity function.
            - 'logistic': The logistic sigmoid function.
            - 'tanh'`, the hyperbolic tan function.
            - 'relu'`, the rectified linear unit function.
            You may also pass a torch module itself, which should be
            callable, taking a tensor as input and outputting a tensor.
            Defaults to :code:`relu`.

        - use_softmax: bool, optional:
            Whether to use a softmax at the end of the fully
            connected layers.
            Defaults to :code:`True`


        '''

        super(FCLayers, self).__init__()

        self.use_softmax = use_softmax
        self.activation = get_function_from_name(activation) if type(activation) == str  \
                            else activation


        in_out_list = [n_input] + list(hidden_layer_sizes) + [n_output] 
        
        in_list = in_out_list[:-1][:-1]
        out_list = in_out_list[:-1][1:]

        self.layers = nn.ModuleList([nn.Sequential(
                                            nn.Linear(in_value, out_value), 
                                            nn.Dropout(dropout),
                                            #nn.BatchNorm1d(out_value), 
                                            )
                                            for in_value, out_value in zip(in_list, out_list)])
        
        self.last_layer = nn.Linear(in_out_list[-2], in_out_list[-1])
        
        if self.use_softmax:
            self.softmax = nn.Softmax(dim=1)

        return
    

    def forward(self, X):
        '''
        Returns
        ---------
            
            out: tensor
                This is the decoded version of the input.
        '''
        
        out = X
        for layer in self.layers:
            out = layer(out)
            out = self.activation(out)
        out = self.last_layer(out)
        if self.use_softmax:
            out = self.softmax(out)

        return out












class MLPModel(BaseLightningModule):
    def __init__(self,
                    n_input:int,
                    n_output:int, 
                    hidden_layer_sizes:typing.List[int]=(100,),
                    activation:typing.Union[str,torch.nn.Module]='relu',
                    use_softmax:bool=True,
                    dropout:float=0.2,
                    optimizer:dict={'adam':{'lr':0.01}},
                    criterion:str='mseloss',
                    n_epochs:int=10,
                    accelerator='auto',
                    **kwargs,
                    ):
        '''
        A simple MLP model that can be used for classification and
        built to be run similar to sklearn models.

        Examples
        ---------
        .. code-block::

            >>> mlp_model = MLPModel(n_input=100, 
            ...     n_output=2, 
            ...     hidden_layer_sizes=(100,100,50),
            ...     n_epochs = 2,
            ...     verbose=True,
            ...     batch_size=10,
            ...     optimizer={'adam':{'lr':0.01}},
            ...     criterion='mseloss',
            ...     )
            >>> X = torch.tensor(np.random.random((10000,100))).float()
            >>> X_val = torch.tensor(np.random.random((10000,100))).float()
            >>> training_metrics = mlp_model.fit(X=X, X_val=X_val)
            >>> output = mlp_model.transform(X_test=X)




        Arguments
        ---------

        - n_input: int:
            The size of the input feature dimension.

        - n_output: int:
            The output dimension sizes.

        - hidden_layer_sizes: typing.List[int], optional:
            The hidden layer sizes.
            Defaults to :code:`(100,)`.

        - activation: typing.Union[str,torch.nn.Module] , optional:
            The activation function to be used in the hidden layers to add
            non-linearity. You may pass a str of the form:
            - 'identity': The identity function.
            - 'logistic': The logistic sigmoid function.
            - 'tanh'`, the hyperbolic tan function.
            - 'relu'`, the rectified linear unit function.
            You may also pass a torch module itself, which should be
            callable, taking a tensor as input and outputting a tensor.
            Defaults to :code:`relu`.

        - use_softmax: bool, optional:
            Whether to use a softmax at the end of the fully
            connected layers.
            Defaults to :code:`True`

        - dropout: float, optional:
            The dropout value in each of the layers.
            Defaults to :code:`0.2`

        - criterion: str or torch.nn.Module:
            The criterion that is used to calculate the loss.
            If using a string, please use one of :code:`['mseloss', 'celoss']`
            Defaults to :code:`mseloss`.

        - optimizer: dict, optional:
            A dictionary containing the optimizer name as keys and
            a dictionary as values containing the arguments as keys. 
            For example: :code:`{'adam':{'lr':0.01}}`. 
            The key can also be a :code:`torch.optim` class,
            but not initiated.
            For example: :code:`{torch.optim.Adam:{'lr':0.01}}`. 
            Defaults to :code:`{'adam':{'lr':0.01}}`.
        
        - n_epochs: int, optional:
            The number of epochs to run the training for.
            Defaults to :code:`10`.
        
        - accelerator: str, optional:
            The device to use for training. Please use 
            any of :code:`(“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “auto”)`.
            Defaults to :code:`'auto'`

        - kwargs: optional:
            These keyword arguments will be passed to 
            :code:`dcarte_transform.model.base_model.BaseModel`.


        '''

        if 'model_name' in kwargs:
            if kwargs['model_name'] is None:
                self.model_name = f'MLP-{n_input}-{n_output}'\
                                    f'-{int("".join(map(str, hidden_layer_sizes)))}-{dropout}'

        super(MLPModel, self).__init__(
            optimizer=optimizer,
            criterion=criterion,
            n_epochs=n_epochs,
            accelerator=accelerator,
            **kwargs,
        )
        
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_layer_sizes = hidden_layer_sizes
        self.use_softmax = use_softmax
        self.dropout = dropout
        self.activation = activation
        self.predict_type = 'classes'

        return
    
    def _build_model(self):
        self.fc_layers = FCLayers(
                            n_input=self.n_input, 
                            n_output=self.n_output, 
                            hidden_layer_sizes=self.hidden_layer_sizes, 
                            use_softmax=self.use_softmax,
                            activation=self.activation,
                            dropout=self.dropout,
                            )
        return

    def forward(self,X):
        return self.fc_layers(X)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.fc_layers(x)
        loss = self.criterion(z, y)
        self.log('train_loss', float(loss))
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.fc_layers(x)
        loss = self.criterion(z, y)
        self.log('val_loss', float(loss), prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx: int):
        if type(batch) == list:
            batch = batch[0]
        batch = batch.view(batch.size(0), -1)
        if self.predict_type == 'classes':
            _, predictions = torch.max(self(batch), dim=1)
            return predictions
        elif self.predict_type == 'probabilities':
            return self(batch)

    def fit(self,
            X:np.array=None, 
            y=None,
            train_loader:torch.utils.data.DataLoader=None,
            X_val:typing.Union[np.array, None]=None,
            y_val=None,
            val_loader:torch.utils.data.DataLoader=None,
            **kwargs,
            ):
        '''
        This is used to fit the model. Please either use 
        the :code:`train_loader` or :code:`X` and :code:`y`.
        This corresponds to using either a torch DataLoader
        or a numpy array as the training data. If using 
        the :code:`train_loader`, ensure each iteration returns
        :code:`[X, X]`.

        Arguments
        ---------

        - X: numpy.array or None, optional:
            The input array to fit the model on.
            Defaults to :code:`None`.

        - train_loader: torch.utils.data.DataLoader or None, optional:
            The training data, which contains the input and the targets.
            Defaults to :code:`None`.

        - X_val: numpy.array or None, optional:
            The validation input to calculate validation 
            loss on when training the model.
            Defaults to :code:`None`

        - val_loader: torch.utils.data.DataLoader or None, optional:
            The validation data, which contains the input and the targets.
            Defaults to :code:`None`.

        '''
        
        self._build_model()

        return super(MLPModel, self).fit(train_loader=train_loader,
                                            X=X, 
                                            y=y,
                                            val_loader=val_loader,
                                            X_val=X_val,
                                            y_val=y_val,
                                            **kwargs,
                                            )
    def predict(self,
                X:np.array=None,
                y:np.array=None,
                test_loader:torch.utils.data.DataLoader=None, 
                ):
        '''
        Method for making predictions on a test loader.
        
        Arguments
        ---------
        
        - X: numpy.array or None, optional:
            The input array to test the model on.
            Defaults to :code:`None`.

        - y: numpy.array or None, optional:
            The target array to test the model on. If set to :code:`None`,
            then :code:`targets_too` will automatically be set to :code:`False`.
            Defaults to :code:`None`.
        
        - test_loader: torch.utils.data.DataLoader or None, optional: 
            A data loader containing the test data.
            Defaults to :code:`None`.
        
        
        Returns
        --------
        
        - output: torch.tensor: 
            The resutls from the predictions
        
        
        '''
        self.predict_type = 'classes'
        return super(MLPModel, self).predict(
                                            X=X, 
                                            y=y,
                                            test_loader=test_loader,
                                            )


    def predict_proba(self,
                X:np.array=None,
                y:np.array=None,
                test_loader:torch.utils.data.DataLoader=None, 
                ):
        '''
        Method for making probability predictions on a test loader.
        
        Arguments
        ---------
        
        - X: numpy.array or None, optional:
            The input array to test the model on.
            Defaults to :code:`None`.

        - y: numpy.array or None, optional:
            The target array to test the model on. If set to :code:`None`,
            then :code:`targets_too` will automatically be set to :code:`False`.
            Defaults to :code:`None`.
        
        - test_loader: torch.utils.data.DataLoader or None, optional: 
            A data loader containing the test data.
            Defaults to :code:`None`.
        
        
        Returns
        --------
        
        - output: torch.tensor: 
            The resutls from the predictions
        
        
        '''
        self.predict_type = 'probabilities'
        return super(MLPModel, self).predict(
                                            X=X, 
                                            y=y,
                                            test_loader=test_loader,
                                            )

