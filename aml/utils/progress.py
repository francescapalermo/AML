import typing
import tqdm
import pandarallel
import numpy as np

# package wide styling for progress bars
tqdm_style = {
                #'ascii':" ▖▘▝▗▚▞▉", 
                'ascii':"▏▎▍▋▊▉", 
                #'colour':'black',
                'dynamic_ncols': True,
                }

