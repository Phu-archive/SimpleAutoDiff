from .scalarOperator import ScalarOperator
import math

class Sigmoid(ScalarOperator):
    def __init__(self, operation1, name="Sigmoid"):
        """
        Getting the value evaluated (operation1.forward()) from operation1
        """
        self.operation1 = operation1
        self.name = name

    def forward(self):
        """
        We try to make the computational graph to be recursive.
        """
        return 1/(1 + math.exp(-self.operation1.forward()))

    def backward(self, grad):
        """
        Call backward on self.operation 1 and 2.
        Given the gradient of "above" operation
        """

        self.operation1.backward(grad * self.forward() * (1 - self.forward()))

    def draw_graph(self, graph):
        graph.node(self.name)
        self.operation1.draw_graph(graph)
        graph.edge(self.operation1.name, self.name)
