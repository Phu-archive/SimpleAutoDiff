from .scalarOperator import ScalarOperator

class Square(ScalarOperator):
    def __init__(self, operation1):
        self.operation1 = operation1

    def forward(self):
        return self.operation1.forward() * self.operation1.forward()

    def backward(self, grad):
        self.operation1.backward(grad * 2 * self.operation1.forward())
