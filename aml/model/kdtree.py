import typing
import scipy
import numpy as np

class KDAttributeTree(scipy.spatial.KDTree):
    def __init__(
        self, 
        data:np.ndarray, 
        *args, 
        attributes:typing.Union[typing.List[np.ndarray], None, np.ndarray]=None, 
        n_jobs:int=1,
        **kwargs,
        ):
        '''
        A wrapper for :code:`scipy.spatial.KDTree` that
        allows you to pass attributes that will be
        returned along with the results from the
        :code:`query` and :code:`query_ball_point` 
        methods. All arguments to the initialisation
        and methods will be passed to :code:`scipy.spatial.KDTree`.
        
        
        Examples
        ---------
        
        .. code-block::
        
            >>> tree = KDAttributeTree(
            ...     data=np.arange(10).reshape(-1,1), 
            ...     attributes=[2*np.arange(10).reshape(-1,1), 3*np.arange(20).reshape(-1,2)],
            ...     )
        
        
        Arguments
        ---------

        - data: np.ndarray:
            The n data points of dimension m to be indexed. 
            This array is not copied unless this is necessary 
            to produce a contiguous array of doubles, and so 
            modifying this data will result in bogus results. 
            The data are also copied if the kd-tree is 
            built with copy_data=True. 
        
        - attributes: typing.Union[typing.List[np.ndarray], None, np.ndarray], optional:
            The list of attributes to be returned with the data.
            If :code:`None`, then all of the outputted 
            attributes will be None. If this is a 1D array,
            it will be transformed to a 2D array. You may also
            pass a list of attributes. Each attribute should be 
            the same length as :code:`data`.
            Defaults to :code:`None`.
        
        - n_jobs: int, optional:
            The number of parallel processes to 
            use when querying the tree.
            Defaults to :code:`1`.
        
        
        '''

        # ensuring that if the attributes is 1d, it is made 2d.
        # then ensuring it is a list
        if attributes is None:
            self.attributes = attributes
        elif type(attributes) == np.ndarray:
            if len(attributes.shape) == 1:
                self.attributes = [attributes.reshape(-1,1)]
            else:
                self.attributes = [attributes]
        elif type(attributes) == list:
            self.attributes = []
            for a in attributes:
                if len(a.shape) == 1:
                    self.attributes.append(a.reshape(-1,1))
                else:
                    self.attributes.append(a)
        else:
            raise TypeError("attributes must be None, np.ndarray, or list of np.ndarrays.")
        
        assert np.all([len(a) == len(data) for a in self.attributes]), \
            "The attributes must have the same length as the data."

        self.n_jobs = n_jobs

        super().__init__(data, *args, **kwargs)

        return
    
    def query(
        self, 
        *args, 
        **kwargs
        )->typing.Tuple[
            typing.Union[float, np.ndarray],
            typing.Union[int, np.ndarray],
            typing.Union[np.ndarray, typing.List[np.ndarray]]
            ]:
        '''
        Query the tree to return the distace, 
        indices, and attributes of the points that
        are the :code:`k` closest to the given data point. 
        
        
        Examples
        ---------
        
        When :code:`k=1` is given with a single point,
        the returned distance and index are floats and integers::
        
            >>> tree.query(5, k=1)
            (0.0, 5, [array([10]), array([30, 33])])
        
        when :code:`k=1` is given with multiple points, 
        the returned distance and indices are 1d arrays::

            >>> tree.query(np.array([[1],[2]]), k=1)
            (array([0., 0.]),
            array([1, 2], dtype=int64),
            [array([[2],
                    [4]]),

            array([[ 6,  9],
                    [12, 15]])])
        
        However, when an array of data points is queried
        with :code:`k` more than 1, the distance and 
        index are returned as 2D arrays::

            >>> tree.query(np.array([[1],[2]]), k=2)
            (array([[0., 1.],
                    [0., 1.]]),
            array([[1, 0],
                    [2, 1]], dtype=int64),
            [array([[[2],
                    [0]],
            
                    [[4],
                    [2]]]),

            array([[[ 6,  9],
                    [ 0,  3]],
            
                    [[12, 15],
                    [ 6,  9]]])])

        Note that if :code:`attributes` was an array in the
        initialisation, the returned attributes will be an 
        array instead of a list.


        Arguments
        ---------

        - *args: 
            All arguments from :code:`scipy.spatial.KDTree` are accepted.
        
        - **kwargs:
            All keyword arguments from 
            :code:`scipy.spatial.KDTree` are accepted.


        Returns
        --------
        
        - d: typing.Union[float, np.ndarray]: 
            The distance of the :code:`k` closest points
            to the queried point.

        - i: typing.Union[int, np.ndarray]: 
            The index of the :code:`k` closest points
            to the queried point.
        
        - a: typing.Union[np.ndarray, typing.List[np.ndarray]]:
            A list of numpy arrays that are the attributes
            of the data points returned in :code:`i`. If
            :code:`attributes` was an array in the
            initialisation, the returned attributes will be an 
            array instead of a list.
        
        '''
        # querying the KDTree
        d, i = super().query(*args, workers=self.n_jobs, **kwargs)

        if self.attributes is None:
            a = self.attributes
        else:
            if len(self.attributes) == 1:
                a = self.attributes[0][i]
            else:
                a = [ax[i] for ax in self.attributes]
        return d, i, a
    
    def query_ball_point(
        self, 
        *args, 
        **kwargs
        )->typing.Tuple[
            typing.List[int],
            typing.Union[
                typing.List[np.ndarray], 
                typing.Union[
                    typing.List[np.ndarray],
                    typing.List[typing.List[np.ndarray]]
                    ]
                ]
            ]:
        '''
        Find all pairs of points between the given
        point and the tree whose distance is at most r.
        It will return the attributes of these points along
        side the index.       
        
        
        Examples
        ---------
        
        The following example is what the query looks
        like on a single point::
        
            >>> tree.query_ball_point(2, r=1)
            ([1, 2, 3],
            [[array([2]), array([6, 9])],
            [array([4]), array([12, 15])],
            [array([6]), array([18, 21])]])
        
        And on multiple points::

            >>> tree.query_ball_point(np.array([[1],[2]]), r=1)
            (array([list([0, 1, 2]), list([1, 2, 3])], dtype=object),
            [[array([[0],
                    [2],
                    [4]]),
            array([[ 0,  3],
                    [ 6,  9],
                    [12, 15]])],

            [array([[2],
                    [4],
                    [6]]),
            array([[ 6,  9],
                    [12, 15],
                    [18, 21]])]])
        
        Arguments
        ---------

        - *args: 
            All arguments from :code:`scipy.spatial.KDTree` are accepted.
        
        - **kwargs:
            All keyword arguments from 
            :code:`scipy.spatial.KDTree` are accepted.
        
        
        Returns
        --------
        
        - d: typing.List[int]: 
            The index of all of the points within 
            :code:`r` of the queried point.

        - a: typing.Union[typing.List[np.ndarray], typing.List[typing.List[np.ndarray]]]: 
            The list of lists of numpy arrays that are the 
            attributes of the points within 
            :code:`r` of the queried point. If
            :code:`attributes` was an array in the
            initialisation, the returned attributes will be a 
            list of arrays instead of a list of lists.
        
        '''
        # querying the KDTree
        i = super().query_ball_point(*args, workers=self.n_jobs, **kwargs)

        if self.attributes is None:
            a = self.attributes
        else:
            if len(self.attributes) == 1:
                a = [self.attributes[0][np.asarray(ix, dtype=int)] for ix in i]
            else:
                a = [[ax[np.asarray(ix, dtype=int)] for ax in self.attributes] for ix in i]
        
        return i, a