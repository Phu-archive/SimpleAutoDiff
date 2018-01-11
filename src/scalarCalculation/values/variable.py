from .baseNumber import BaseNumber

class Variables(BaseNumber):
    """
    This is going to represent all the variables that able to change its values
    and calculate gradients.
    """
    def __init__(self, value):
        super().__init__(value)
        self.grad = 1
