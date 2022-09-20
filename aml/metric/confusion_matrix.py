import typing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_confusion_matrix(
    cfm:np.ndarray,
    group_names:typing.Union[None, typing.List[str]]=None,
    categories:typing.Union[None, typing.List[str]]='auto',
    count:bool=True,
    percent:bool=True,
    cbar:bool=True,
    xyticks:bool=True,
    xyplotlabels:bool=True,
    sum_stats:bool=True,
    figsize:typing.Union[tuple, list]=None,
    cmap:str='Blues',
    title:typing.Union[str, None]=None,
    ) -> typing.Tuple[plt.figure, plt.axes]:
    '''
    This function was edited from :code:`https://github.com/DTrimarchi10/confusion_matrix/blob/master/cfm_matrix.py`.

    This function will plot a confusion matrix, based on the array given.


    Examples
    ---------

    .. code-block::

        >>> from sklearn.metrics import confusion_matrix
        >>> make_confusion_matrix(
            confusion_matrix(
                y_test, 
                predictions_test
                )
            )


    Arguments
    ---------
    

    - cfm: np.ndarray: 
        Confusion matrix to be plotted.
    
    - group_names: typing.Union[None, typing.List[str]], optional:
        List of strings that represent the labels row by row to be shown in each square. 
        Defaults to :code:`None`.
    
    - categories: typing.Union[None, typing.List[str]], optional:
        List of strings containing the categories to be displayed on the x,y axis. 
        Defaults to :code:`'auto'`.
    
    - count: bool, optional:
        If True, show the raw number in the confusion matrix. 
        Defaults to :code:`True`.
    
    - percent: bool, optional:
        If True, show the proportions for each category. 
        Defaults to :code:`True`.
    
    - cbar: bool, optional:
        If True, show the color bar. The cbar values are based off the values in the confusion matrix. 
        Defaults to :code:`True`.
    
    - xyticks: bool, optional:
        If True, show x and y ticks. 
        Defaults to :code:`True`.
    
    - xyplotlabels: bool, optional:
        If True, show 'True Label' and 'Predicted Label' on the figure. 
        Defaults to :code:`True`.
    
    - sum_stats: bool, optional:
        If True, display summary statistics below the figure. 
        Defaults to :code:`True`.
    
    - figsize: typing.Union[tuple, list], optional:
        Tuple representing the figure size.  If :code:`None`, the matplotlib
        rcParams value will be used.
        Defaults to :code:`None`.
    
    - cmap: str, optional:
        Colormap of the values displayed from matplotlib.pyplot.cm. 
        See :code:`http://matplotlib.org/examples/color/colormaps_reference.html`.
        Defaults to :code:`'Blues'`.
    
    - title: typing.Union[str, None], optional:
        Title for the heatmap. 
        Defaults to :code:`None`.
    
    
    Returns
    ---------

    - figure: matplotlib.pyplot.figure:
        The figure containing the axes.
    
    - ax: matplotlib.pyplot.axes:
        The axes containing the plot.

    
    '''



    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cfm.size)]

    if group_names and len(group_names)==cfm.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cfm.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cfm.flatten()/np.sum(cfm)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cfm.shape[0],cfm.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cfm) / float(np.sum(cfm))

        #if it is a binary confusion matrix, show some more stats
        if len(cfm)==2:
            #Metrics for Binary Confusion Matrices
            precision = cfm[1,1] / sum(cfm[:,1])
            recall    = cfm[1,1] / sum(cfm[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax=sns.heatmap(
        cfm,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
        ax=ax,
        )

    if xyplotlabels:
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label' + stats_text)
    else:
        ax.set_xlabel(stats_text)
    
    if title:
        ax.set_title(title)
    
    return fig, ax