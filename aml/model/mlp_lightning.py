import torch
from torch import nn
import numpy as np
import typing
from copy import deepcopy

from .base_model import BaseLightningModule




class FCLayers(nn.Module):
    def __init__(self, 
                    n_input:int, 
                    n_output:int, 
                    hidden_layer_sizes:typing.List[int]=(100,),
                    use_softmax:bool=True, 
                    dropout=0.2,
                    ):
        '''
        This class can be used on its own as an MLP.

        Arguments
        ---------
            
        - ```n_input```: ```int```:
            The size of the input feature dimension.

        - ```n_output```: ```int```:
            The output dimension sizes.

        - ```hidden_layer_sizes```: ```typing.List[int]```, optional:
            The hidden layer sizes.
            Defaults to ```(100,)```.

        - ```use_softmax```: ```bool```, optional:
            Whether to use a softmax at the end of the fully
            connected layers.
            Defaults to ```True```


        '''

        super(FCLayers, self).__init__()

        self.use_softmax = use_softmax
        
        in_out_list = [n_input] + list(hidden_layer_sizes) + [n_output] 
        
        in_list = in_out_list[:-1][:-1]
        out_list = in_out_list[:-1][1:]

        self.layers = nn.ModuleList([nn.Sequential(
                                            nn.Linear(in_value, out_value), 
                                            nn.Dropout(dropout),
                                            nn.BatchNorm1d(out_value), 
                                            nn.ReLU(),
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
        out = self.last_layer(out)
        if self.use_softmax:
            out = self.softmax(out)

        return out












class MLPModel(BaseLightningModule):
    def __init__(self,
                    n_input:int,
                    n_output:int, 
                    hidden_layer_sizes:typing.List[int]=(100,),
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

        Example
        ---------
        ```
        mlp_model = MLPModel(n_input=100, 
                            n_output=2, 
                            hidden_layer_sizes=(100,100,50),
                            n_epochs = 2,
                            verbose=True,
                            batch_size=10,
                            optimizer={'adam':{'lr':0.01}},
                            criterion='mseloss',
                            )

        X = torch.tensor(np.random.random((10000,100))).float()
        X_val = torch.tensor(np.random.random((10000,100))).float()

        training_metrics = mlp_model.fit(X=X, X_val=X_val)
        output = mlp_model.transform(X_test=X)


        ```


        Arguments
        ---------

        - ```n_input```: ```int```:
            The size of the input feature dimension.

        - ```n_output```: ```int```:
            The output dimension sizes.

        - ```hidden_layer_sizes```: ```typing.List[int]```, optional:
            The hidden layer sizes.
            Defaults to ```(100,)```.

        - ```use_softmax```: ```bool```, optional:
            Whether to use a softmax at the end of the fully
            connected layers.
            Defaults to ```True```

        - ```dropout```: ```float```, optional:
            The dropout value in each of the layers.
            Defaults to ```0.2```

        - ```criterion```: ```str``` or ```torch.nn.Module```:
            The criterion that is used to calculate the loss.
            If using a string, please use one of ```['mseloss', 'celoss']```
            Defaults to ```mseloss```.

        - ```optimizer```: ```dict```, optional:
            A dictionary containing the optimizer name as keys and
            a dictionary as values containing the arguments as keys. 
            For example: ```{'adam':{'lr':0.01}}```. 
            The key can also be a ```torch.optim``` class,
            but not initiated.
            For example: ```{torch.optim.Adam:{'lr':0.01}}```. 
            Defaults to ```{'adam':{'lr':0.01}}```.
        
        - ```n_epochs```: ```int```, optional:
            The number of epochs to run the training for.
            Defaults to ```10```.
        
        - ```accelerator```: ```str```, optional:
            The device to use for training. Please use 
            any of ```(“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “auto”)```.
            Defaults to ```'auto'```

        - ```kwargs```: optional:
            These keyword arguments will be passed to 
            ```dcarte_transform.model.base_model.BaseModel```.


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

        return
    
    def _build_model(self):
        self.fc_layers = FCLayers(
                                    n_input=self.n_input, 
                                    n_output=self.n_output, 
                                    hidden_layer_sizes=self.hidden_layer_sizes, 
                                    use_softmax=self.use_softmax,
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
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.fc_layers(x)
        loss = self.criterion(z, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx: int):
        if type(batch) == list:
            batch = batch[0]
        batch = batch.view(batch.size(0), -1)
        _, predictions = torch.max(self(batch), dim=1)
        return predictions

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
        the ```train_loader``` or ```X``` and ```y```.
        This corresponds to using either a torch DataLoader
        or a numpy array as the training data. If using 
        the ```train_loader```, ensure each iteration returns
        ```[X, X]```.

        Arguments
        ---------

        - ```X```: ```numpy.array``` or ```None```, optional:
            The input array to fit the model on.
            Defaults to ```None```.

        - ```train_loader```: ```torch.utils.data.DataLoader``` or ```None```, optional:
            The training data, which contains the input and the targets.
            Defaults to ```None```.

        - ```X_val```: ```numpy.array``` or ```None```, optional:
            The validation input to calculate validation 
            loss on when training the model.
            Defaults to ```None```

        - ```val_loader```: ```torch.utils.data.DataLoader``` or ```None```, optional:
            The validation data, which contains the input and the targets.
            Defaults to ```None```.

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

