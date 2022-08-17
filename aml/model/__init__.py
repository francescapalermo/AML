from .autoencoder_lightning import AEModel
from .mlp_lightning import MLPModel

import warnings
import logging 

## pytorch lightning warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", '.*You defined a `validation_step` but have no `val_dataloader`.*')
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


__all__=[
        'AEModel',
        'MLPModel',
        ]