from .baseNumber import BaseNumber
import warnings

class Constant(BaseNumber):
    """
    Constant can't be change after initialized.
    """

    def __init__(self, value):
        super().__init__(value)
        if not (type(value) == int or type(value) == float):
            raise TypeError("Constant has to be int or float!!")

    def get_grad(self):
        """
        For people who accidently call get the gradient from the constant
        """

        warnings.warn("get_grad for constant will always be zero!!")
        return 0

    def backward(self, grad):
        # just ignore the data !!!
        pass
