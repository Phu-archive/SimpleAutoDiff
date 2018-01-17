from .baseNumber import BaseNumber

class Variables(BaseNumber):
    """
    This is going to represent all the variables that able to change its values
    and calculate gradients.
    """
    def __init__(self, value, name="Variable"):
        super().__init__(value, name)
        if not(type(self.value) == int or type(self.value) == float):
            raise TypeError("Variable has to be int or float!!")
        self.grad = 1

    def backward(self, grad):
        # Just store the grad data for further use.
        self.grad = grad

    def get_grad(self):
        return self.grad

    def draw_graph(self, graph):
        pass
