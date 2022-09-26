from joblib import Parallel
import tqdm

class ProgressParallel(Parallel):
    def __init__(self, tqdm_bar:tqdm.tqdm, *args, **kwargs):
        '''
        This is a wrapper for the joblib Parallel
        class that allows for a progress bar to be passed into
        the :code:`__init__` function so that the progress 
        can be viewed.
        
        
        
        Examples
        ---------
        
        .. code-block:: 
        
            >>> pbar = tqdm.tqdm(total=5)
            >>> result = ProgressParallel(
                    tqdm_bar=pbar,
                    n_jobs=10,
                    )(
                        joblib.delayed(f_parallel)(i)
                        for i in range(5)
                    )
        
        
        Arguments
        ---------
        
        - tqdm_bar: tqdm.tqdm: 
            The tqdm bar that will be used in the
            progress updates.
            Every time progress is displayed, 
            :code:`tqdm_bar.update(n)` will be called,
            where :code:`n` is the number of updates made.

        
        '''
        self.tqdm_bar = tqdm_bar
        self.previously_completed = 0
        super().__init__(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        difference = self.n_completed_tasks - self.previously_completed
        self.tqdm_bar.update(difference)
        self.tqdm_bar.refresh()
        self.previously_completed += difference