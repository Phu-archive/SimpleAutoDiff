from .scalarOperator import ScalarOperator

class Square(ScalarOperator):
    def __init__(self, operation1, name="Square"):
        self.operation1 = operation1
        self.name = name

    def forward(self):
        return self.operation1.forward() * self.operation1.forward()

    def backward(self, grad):
        self.operation1.backward(grad * 2 * self.operation1.forward())

    def draw_graph(self, graph):
        graph.node(self.name)
        self.operation1.draw_graph(graph)
        graph.edge(self.operation1.name, self.name)
