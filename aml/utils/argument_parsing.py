import typing

class ArgFake:
    def __init__(self, **arguments:typing.Any):
        '''
        Fake argparse arguments
        
        Arguments
        ---------
        
        - arguments: typing.Any: 
            Keyword arguments that will be accessable as
            attributes of this class.
        
        '''
        self.arguments = arguments
        for key in self.arguments:
            setattr(self, key, self.arguments[key])

    def __str__(self):
        return str(self.arguments)
