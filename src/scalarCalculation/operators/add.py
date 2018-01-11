from .scalarOperator import ScalarOperator

class Add(ScalarOperator):
    """
    Adding together the same scalar types (constant and variable)
    """
    def __init__(self, operation1, operation2):
        """
        Getting the left value evaluated (operation1.forward()) from operation1
        and add to right value evaluated (operation2.forward()) from operation2
        """
        self.operation1 = operation1
        self.operation2 = operation2

    def forward(self):
        """
        We try to make the computational graph to be recursive.
        """
        return operation1.forward() + scalar2.forward()

    def backward(self):
        pass
