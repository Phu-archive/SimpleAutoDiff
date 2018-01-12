from abc import ABC, abstractmethod

class ScalarOperator(ABC):
    """
    Base class for all operators that uses scalar as input.
    """
    @abstractmethod
    def forward (self):
        pass

    @abstractmethod
    def backward(self, grad):
        pass
