import graphviz as gv


def plot_graph(computational_graph, output_file_name="img/graph", file_type="svg", mode="s"):
    graph_diagram = gv.Digraph(format=file_type)
    computational_graph.draw_graph(graph_diagram)

    # Mode save
    if mode == "s":
        print("Finish you can see the graph at,", output_file_name)
        graph_diagram.render(output_file_name)

    # Mode draw
    elif mode == "d":
        return graph_diagram
