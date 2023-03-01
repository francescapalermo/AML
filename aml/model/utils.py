import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


def get_optimizer_from_name(name):
    """
    Get optimizer from name.



    Arguments
    ---------

    - name: str:
        This can be any of :code:`'adam'`,
        :code:`'adadelta'`, :code:`'sgd'`


    Raises
    ---------

        :code:`NotImplementedError: If optimizer name
        is not implemented.

    Returns
    --------

    - optimizer: torch.optim` optimizer.


    """
    if name == "adam":
        return torch.optim.Adam
    elif name == "adadelta":
        return torch.optim.Adadelta
    elif name == "sgd":
        return torch.optim.SGD
    else:
        raise NotImplementedError(
            "Sorry, {} is not a valid optimizer name.".format(name)
        )


def get_criterion_from_name(name):
    """
    Get loss function from name.



    Arguments
    ---------

    - name: str:
        This can be any of :code:`'celoss'`,
        :code:`'mseloss'`.


    Raises
    ---------

        :code:`NotImplementedError: If loss name
        is not implemented.

    Returns
    --------

    - loss_function`.


    """
    if name == "celoss":
        return nn.CrossEntropyLoss()
    elif name == "mseloss":
        return nn.MSELoss()


def get_function_from_name(name):
    """
    Get a torch.nn.functional from a given name.



    Arguments
    ---------

    - name: str:
        This can be any of:
        - 'identity': The identity function.
        - 'logistic'` or :code:`'sigmoid': The logistic sigmoid function.
        - 'tanh'`, the hyperbolic tan function.
        - 'relu'`, the rectified linear unit function.


    Raises
    ---------

        :code:`NotImplementedError: If loss name
        is not implemented.

    Returns
    --------

    - loss_function`.


    """

    if name == "relu":
        return F.relu
    elif name == "logistic" or "sigmoid":
        return torch.sigmoid
    elif name == "tanh":
        return F.tanh
    elif name == "identity":
        return lambda x: x
