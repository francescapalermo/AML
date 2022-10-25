import numpy as np
import typing
from sklearn.metrics._classification import (
    _check_set_wise_labels,
    multilabel_confusion_matrix,
    _prf_divide
    )



def countna(array:np.ndarray, normalise:bool=True,) -> float:
    '''
    Function to calculate the number of NAs in
    an array.

    Arguments
    ---------

    - array: numpy.ndarray:
        The array to calculate the number of missing
        values on.
    
    - normalise: bool, optional:
        Whether to return the values as a percentage.
        Defaults to :code:`True`.
    
    Returns
    ---------

    - countna: float:
        A :code:`float` equal to the number or proportion
        of missing values in an array.

    '''
    count_na = np.count_nonzero(np.isnan(array))
    if normalise:
        count_na *= 1/array.shape[0]
    return count_na


def interquartile_range(values:np.ndarray, lower:float=25, upper:float=75) -> float:
    '''
    Function to calculate the interquartile
    range of an array.

    Arguments
    ---------

    - array: numpy.ndarray:
        The array to calculate the IQR of.
    
    - lower: float, optional:
        The percentile of the lower quartile.
        Defaults to :code:`25`.
    
    - upper: float, optional:
        The percentile of the upper quartile.
        Defaults to :code:`75`.

    Returns
    ---------

    - iqr: float:
        A :code:`float` equal to the interquartile range.

    '''
    return np.subtract(*np.nanpercentile(values, [upper, lower]))


def format_mean_iqr_missing(
    values:np.ndarray, 
    string:str="{mean:.2f} ({iqr:.2f}) (({count_na:.0f}%))",
    ) -> str:
    '''
    A function useful for formatting a table with information 
    on the mean, IQR and missing rate of an attribute.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> format_mean_iqr_missing(
                values=np.array([1,2,3,4,5]),
                string="{mean:.2f} ({iqr:.2f}) (({count_na:.0f}%))",
                )
        '3.00 (2) ((0%))'


    Arguments
    ---------

    - values: numpy.ndarray:
        The array to calculate the values on.
    
    - string: str, optional:
        The string that dictates the output.
        This should include somewhere :code:`{mean}`,
        :code:`{iqr}`, and :code:`{count_na}`.
        Defaults to :code:`"{mean:.2f} ({iqr:.2f}) (({count_na:.0f}%))"`.
    
    Returns
    ---------

    - stats: str:
        A string of the desired format with the 
        statistics included.

    
    '''
    mean = np.mean(values)
    iqr = interquartile_range(values)
    count_na = countna(values)*100
    return string.format(mean=mean, iqr=iqr, count_na=count_na)



def format_mean_std(
    values:np.ndarray, 
    string:str="{mean:.2f} ({std:.2f})",
    ) -> str:
    '''
    A function useful for formatting a table with information 
    on the mean and standard deviation an attribute.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> format_mean_std(
                values=np.array([1,2,3,4,5]),
                string="{mean:.2f} ({std:.2f})",
                )
        '3.00 (1.41)'


    Arguments
    ---------

    - values: numpy.ndarray:
        The array to calculate the values on.
    
    - string: str, optional:
        The string that dictates the output.
        This should include somewhere :code:`{mean}` and
        :code:`{std}`.
        Defaults to :code:`"{mean:.2f} ({std:.2f})"`.
    
    Returns
    ---------

    - stats: str:
        A string of the desired format with the 
        statistics included.

    
    '''
    mean = np.mean(values)
    std = np.std(values)
    return string.format(mean=mean, std=std)






def sensitivity_specificity(
    y_true:np.ndarray, 
    y_pred:np.ndarray, 
    labels:typing.Union[np.ndarray, None]=None,
    pos_label:typing.Union[str, int]=1,
    average:typing.Union[str, None]=None,
    warn_for:typing.Union[str, typing.Tuple[str]]=("sensitivity", "specificity"),
    sample_weight:typing.Union[np.ndarray, None]=None,
    zero_division:typing.Union[int, str]='warn',
    ) -> typing.Tuple[float]:
    '''
    A function that calculates the sensitivity and
    specificity between two arrays. This is modelled 
    on the Scikit-Learn :code:`recall_score`, 
    :code:`precision_score`, :code:`accuracy_score`,
    and :code:`f1_score`.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> sensitivity_specificity(
                y_true=np.array([0,1,0,1,0]),
                y_pred=np.array([0,0,0,1,0]),
                )
        (0.5, 1.0)


    Arguments
    ---------

    - y_true: numpy.ndarray:
        The array of true values.

    - y_pred: numpy.ndarray:
        The array of predicted values.
    
    - labels: typing.Union[np.ndarray, None], optional:
        The set of labels to include when 
        :code:`average != 'binary'`, and their order 
        if average is :code:`None`. Labels present in 
        the data can be excluded, for example to calculate a 
        multiclass average ignoring a majority 
        negative class, while labels not present 
        in the data will result in 0 components in a 
        macro average. For multilabel targets, labels are 
        column indices. By default, all labels in 
        :code:`y_true` and :code:`y_pred` are used in sorted order.
        Defaults to :code:`None`.

    - pos_label: typing.Union[str, int], optional:
        The class to report if :code:`average='binary'` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting :code:`labels=[pos_label]` and 
        :code:`average != 'binary'` will report
        scores for that label only.
        Defaults to :code:`1`.

    - average: typing.Union[str, None], optional:
        If :code:`None`, the scores for each class 
        are returned. Otherwise, this determines the type 
        of averaging performed on the data:
        - :code:`'binary'`:
            Only report results for the class specified by :code:`pos_label`.
            This is applicable only if targets (`y_{true,pred}`) are binary.
        - :code:`'micro'`:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        - :code:`'macro'`:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        - :code:`'weighted'`:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        - :code:`'samples'`:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :code:`accuracy_score`).
        Defaults to :code:`None`.

    - warn_for: typing.Union[str, typing.Tuple[str]], optional:
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.
        Defaults to :code:`("sensitivity", "specificity")`.

    - sample_weight: typing.Union[np.ndarray, None], optional:
        Sample weights.
        Defualts to :code:`None`.

    - zero_division: typing.Union[int, str], optional:
        Sets the value to return when there is a zero division:
           - sensitivity: when there are no positive labels
           - specificity: when there are no negative labels
        If set to :code:`"warn"`, this acts as :code:`0`, 
        but warnings are also raised.
        Defaults to :code:`"warn"`.
    
    Returns
    ---------

    - sensitivity: float:
        The sensitivity score.

    - specificity: float:
        The specificity score.
    
    '''
    
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)

    # Calculate tp_sum, pred_sum, true_sum ###
    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=labels,
        samplewise=samplewise,
    )

    tp_sum = MCM[:, 1, 1]
    tn_sum =  MCM[:, 0, 0]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]
    false_sum = tn_sum + MCM[:, 0, 1]

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        tn_sum = np.array([tn_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])
        false_sum = np.array([false_sum.sum()])

    sensitivity = _prf_divide(
        tp_sum, 
        true_sum, 
        "sensitivity", 
        "true", 
        average, 
        warn_for, 
        zero_division=zero_division,
    )

    specificity = _prf_divide(
        tn_sum, 
        false_sum, 
        "specificity", 
        "true", 
        average, 
        warn_for, 
        zero_division=zero_division,
    )

    sensitivity_zero_division = False
    specificity_zero_division = False

    if average == "weighted":
        weights_sensitivity = true_sum
        weights_specificity = false_sum

        if weights_sensitivity.sum() == 0:
            zero_division_value = np.float64(1.0)
            if zero_division in ["warn", 0]:
                zero_division_value = np.float64(0.0)
            # sensitivity is zero_division if there are no positive labels
            sensitivity = zero_division_value
            sensitivity_zero_division = True

        if weights_specificity.sum() == 0:
            zero_division_value = np.float64(1.0)
            if zero_division in ["warn", 0]:
                zero_division_value = np.float64(0.0)
            # specificity is zero_division if there are no negative labels
            specificity = zero_division_value
            specificity_zero_division = True


    elif average == "samples":
        weights_sensitivity = sample_weight
        weights_specificity = sample_weight
    else:
        weights_sensitivity = None
        weights_specificity = None

    if average is not None:
        assert average != "binary" or len(sensitivity) == 1
        if not sensitivity_zero_division:
            sensitivity = np.average(sensitivity, weights=weights_sensitivity)
        if not specificity_zero_division:
            specificity = np.average(specificity, weights=weights_specificity)

    return sensitivity, specificity




def sensitivity_score(
    y_true:np.ndarray, 
    y_pred:np.ndarray, 
    labels:typing.Union[np.ndarray, None]=None,
    pos_label:typing.Union[str, int]=1,
    average:typing.Union[str, None]="binary",
    sample_weight:typing.Union[np.ndarray, None]=None,
    zero_division:typing.Union[int, str]='warn',
    ) -> float:
    '''
    A function that calculates the sensitivity
    between two arrays. This is modelled 
    on the Scikit-Learn :code:`recall_score`, 
    :code:`precision_score`, :code:`accuracy_score`,
    and :code:`f1_score`.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> sensitivity_score(
                y_true=np.array([0,1,0,1,0]),
                y_pred=np.array([0,0,0,1,0]),
                )
        0.5


    Arguments
    ---------

    - y_true: numpy.ndarray:
        The array of true values.

    - y_pred: numpy.ndarray:
        The array of predicted values.
    
    - labels: typing.Union[np.ndarray, None], optional:
        The set of labels to include when 
        :code:`average != 'binary'`, and their order 
        if average is :code:`None`. Labels present in 
        the data can be excluded, for example to calculate a 
        multiclass average ignoring a majority 
        negative class, while labels not present 
        in the data will result in 0 components in a 
        macro average. For multilabel targets, labels are 
        column indices. By default, all labels in 
        :code:`y_true` and :code:`y_pred` are used in sorted order.
        Defaults to :code:`None`.

    - pos_label: typing.Union[str, int], optional:
        The class to report if :code:`average='binary'` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting :code:`labels=[pos_label]` and 
        :code:`average != 'binary'` will report
        scores for that label only.
        Defaults to :code:`1`.

    - average: typing.Union[str, None], optional:
        If :code:`None`, the scores for each class 
        are returned. Otherwise, this determines the type 
        of averaging performed on the data:
        - :code:`'binary'`:
            Only report results for the class specified by :code:`pos_label`.
            This is applicable only if targets (`y_{true,pred}`) are binary.
        - :code:`'micro'`:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        - :code:`'macro'`:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        - :code:`'weighted'`:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        - :code:`'samples'`:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :code:`accuracy_score`).
        Defaults to :code:`"binary"`.

    - sample_weight: typing.Union[np.ndarray, None], optional:
        Sample weights.
        Defualts to :code:`None`.

    - zero_division: typing.Union[int, str], optional:
        Sets the value to return when there is a zero division:
           - sensitivity: when there are no positive labels
           - specificity: when there are no negative labels
        If set to :code:`"warn"`, this acts as :code:`0`, 
        but warnings are also raised.
        Defaults to :code:`"warn"`.
    
    Returns
    ---------

    - sensitivity: float:
        The sensitivity score.
    
    '''

    warn_for = "sensitivity",
    s, _ = sensitivity_specificity(
        y_true=y_true, 
        y_pred=y_pred, 
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        warn_for=warn_for,
        zero_division=zero_division,
    )
    return s





def specificity_score(
    y_true:np.ndarray, 
    y_pred:np.ndarray, 
    labels:typing.Union[np.ndarray, None]=None,
    pos_label:typing.Union[str, int]=1,
    average:typing.Union[str, None]="binary",
    sample_weight:typing.Union[np.ndarray, None]=None,
    zero_division:typing.Union[int, str]='warn',
    ):
    '''
    A function that calculates the sensitivity
    between two arrays. This is modelled 
    on the Scikit-Learn :code:`recall_score`, 
    :code:`precision_score`, :code:`accuracy_score`,
    and :code:`f1_score`.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> specificity_score(
                y_true=np.array([0,1,0,1,0]),
                y_pred=np.array([0,0,0,1,0]),
                )
        1.0


    Arguments
    ---------

    - y_true: numpy.ndarray:
        The array of true values.

    - y_pred: numpy.ndarray:
        The array of predicted values.
    
    - labels: typing.Union[np.ndarray, None], optional:
        The set of labels to include when 
        :code:`average != 'binary'`, and their order 
        if average is :code:`None`. Labels present in 
        the data can be excluded, for example to calculate a 
        multiclass average ignoring a majority 
        negative class, while labels not present 
        in the data will result in 0 components in a 
        macro average. For multilabel targets, labels are 
        column indices. By default, all labels in 
        :code:`y_true` and :code:`y_pred` are used in sorted order.
        Defaults to :code:`None`.

    - pos_label: typing.Union[str, int], optional:
        The class to report if :code:`average='binary'` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting :code:`labels=[pos_label]` and 
        :code:`average != 'binary'` will report
        scores for that label only.
        Defaults to :code:`1`.

    - average: typing.Union[str, None], optional:
        If :code:`None`, the scores for each class 
        are returned. Otherwise, this determines the type 
        of averaging performed on the data:
        - :code:`'binary'`:
            Only report results for the class specified by :code:`pos_label`.
            This is applicable only if targets (`y_{true,pred}`) are binary.
        - :code:`'micro'`:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        - :code:`'macro'`:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        - :code:`'weighted'`:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        - :code:`'samples'`:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :code:`accuracy_score`).
        Defaults to :code:`"binary"`.

    - sample_weight: typing.Union[np.ndarray, None], optional:
        Sample weights.
        Defualts to :code:`None`.

    - zero_division: typing.Union[int, str], optional:
        Sets the value to return when there is a zero division:
           - sensitivity: when there are no positive labels
           - specificity: when there are no negative labels
        If set to :code:`"warn"`, this acts as :code:`0`, 
        but warnings are also raised.
        Defaults to :code:`"warn"`.
    
    Returns
    ---------

    - sensitivity: float:
        The sensitivity score.
    
    '''

    warn_for = "specificity",
    _, s = sensitivity_specificity(
        y_true=y_true, 
        y_pred=y_pred, 
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        warn_for=warn_for,
        zero_division=zero_division,
    )
    return s