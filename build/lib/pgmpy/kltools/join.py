from pgmpy.base import DirectedGraph
from pgmpy.base import UndirectedGraph


def joinDirectedGraph(first, second):
    """
    Returns a new undirected graph with all the edges from moral
    graphs of self and other
    Parameters
    ----------
    other: graph to join with self

    Examples
    --------
    >>> from pgmpy.base import DirectedGraph
    >>> f = DirectedGraph(ebunch=[('A', 'B'), ('C', 'B'), ('B', 'D')])

    # defines g graph
    >>> g = DirectedGraph(ebunch=[('A', 'B'), ('A', 'C'), ('C', 'D')])
    >>> joinDirectedGraph(f, g)
    [('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'C'), ('C', 'D')])

    """
    # moralize both graphs
    firstMoral = first.moralize()
    secondMoral = second.moralize()

    # join both of them as undirected graphs
    return joinUndirectedGraph(firstMoral, secondMoral)


def joinUndirectedGraph(first, second):
    """
    Returns a new undirected graph with all the edges from moral
    graphs of self and other
    Parameters
    ----------
    other: graph to join with self

    Examples
    --------
    >>> from pgmpy.base import UndirectedGraph
    >>> f = UndirectedGraph(ebunch=[('A', 'B'), ('C', 'B'), ('B', 'D')])

    # defines g graph
    >>> g = UndirectedGraph(ebunch=[('A', 'B'), ('A', 'C'), ('C', 'D')])
    >>> joinUndirectedGraph(f, g)
    [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'D')])

    """
    result = UndirectedGraph(ebunch=first.edges())

    # adds other edges
    for edge in second.edges():
        if edge not in first.edges():
            result.add_edge(u=edge[0], v=edge[1])

    # return result
    return result
