from .scalarOperator import ScalarOperator

class Add(ScalarOperator):
    """
    Adding together the same scalar types (constant and variable)
    """
    def __init__(self, operation1, operation2, name="Add"):
        """
        Getting the left value evaluated (operation1.forward()) from operation1
        and add to right value evaluated (operation2.forward()) from operation2
        """
        self.operation1 = operation1
        self.operation2 = operation2
        self.name = name

    def forward(self):
        """
        We try to make the computational graph to be recursive.
        """
        return self.operation1.forward() + self.operation2.forward()

    def backward(self, grad):
        """
        Call backward on self.operation 1 and 2.
        Given the gradient of "above" operation
        """

        self.operation1.backward(grad)
        self.operation2.backward(grad)

    def draw_graph(self, graph):
        graph.node(self.name)

        self.operation1.draw_graph(graph)
        self.operation2.draw_graph(graph)

        graph.edge(self.operation1.name, self.name)
        graph.edge(self.operation2.name, self.name)


class Minus(ScalarOperator):
    """
    Adding together the same scalar types (constant and variable)
    """
    def __init__(self, operation1, operation2, name="Minus"):
        """
        Getting the left value evaluated (operation1.forward()) from operation1
        and add to right value evaluated (operation2.forward()) from operation2
        """
        self.operation1 = operation1
        self.operation2 = operation2
        self.name = name

    def forward(self):
        """
        We try to make the computational graph to be recursive.
        """
        return self.operation1.forward() - self.operation2.forward()

    def backward(self, grad):
        """
        Call backward on self.operation 1 and 2.
        Given the gradient of "above" operation
        """

        self.operation1.backward(grad)
        self.operation2.backward(grad*-1)

    def draw_graph(self, graph):
        graph.node(self.name)

        self.operation1.draw_graph(graph)
        self.operation2.draw_graph(graph)

        graph.edge(self.operation1.name, self.name)
        graph.edge(self.operation2.name, self.name)


class Multiply(ScalarOperator):
    def __init__(self, operation1, operation2, name="Multiply"):
        """
        Getting the left value evaluated (operation1.forward()) from operation1
        and add to right value evaluated (operation2.forward()) from operation2
        """
        self.operation1 = operation1
        self.operation2 = operation2
        self.name = name

    def forward(self):
        """
        We try to make the computational graph to be recursive.
        """
        return self.operation1.forward() * self.operation2.forward()

    def backward(self, grad):
        self.operation1.backward(self.operation2.forward()*grad)
        self.operation2.backward(self.operation1.forward()*grad)

    def draw_graph(self, graph):
        graph.node(self.name)

        self.operation1.draw_graph(graph)
        self.operation2.draw_graph(graph)

        graph.edge(self.operation1.name, self.name)
        graph.edge(self.operation2.name, self.name)
