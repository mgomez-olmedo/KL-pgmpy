from pgmpy.base import DirectedGraph
from pgmpy.base import UndirectedGraph
from pgmpy.kltools.join import joinDirectedGraph
from pgmpy.kltools.join import joinUndirectedGraph

# defines f graph
f = DirectedGraph(ebunch=[('A', 'B'), ('C', 'B'), ('B', 'D')])
print("f graph edges: ", f.edges())

# defines g graph
g = DirectedGraph(ebunch=[('A', 'B'), ('A', 'C'), ('C', 'D')])
print("g graph edges: ", g.edges())

# now call join method
joined = joinDirectedGraph(f, g)
print("joined graph edges: ", joined.edges())

# proof with undireected graph
from pgmpy.base import UndirectedGraph

f = UndirectedGraph(ebunch=[('A', 'B'), ('C', 'B'), ('B', 'D')])
g = UndirectedGraph(ebunch=[('A', 'B'), ('A', 'C'), ('C', 'D')])
joined = joinUndirectedGraph(f, g)
print("joined graph edges: ", joined.edges())