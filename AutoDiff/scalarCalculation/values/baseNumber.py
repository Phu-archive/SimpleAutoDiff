from abc import ABC, abstractmethod

class BaseNumber(ABC):
    """
    Base Class that represent the constant numbers and variables.
    """
    def __init__(self, value):
        self.value = value

    def forward(self):
        """
        When you forward a value, you get that value.
        """
        return self.value

    def backward(self, grad):
        pass

    def get_grad(self):
        pass
