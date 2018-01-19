from abc import ABC, abstractmethod

class BaseNumber(ABC):
    """
    Base Class that represent the constant numbers and variables.
    """
    def __init__(self, value, name):
        self.value = value
        self.name = name

    def forward(self):
        """
        When you forward a value, you get that value.
        """
        return self.value

    def backward(self, grad):
        pass

    def get_grad(self):
        pass

    @abstractmethod
    def draw_graph(self, graph):
        pass
