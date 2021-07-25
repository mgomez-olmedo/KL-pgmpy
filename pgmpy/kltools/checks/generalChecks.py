from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.utils import get_example_model

############################################################
# code for learning how to manage model cpds
############################################################

# read model for asia
model = get_example_model('cancer')

# get access to all cpds
cpds = model.cpds
print(cpds)

# get a cpd and see how to loop over its values
nodes = model.nodes()
for node in nodes:
    cpd  = model.get_cpds(node)
    print("cpd for node: ", node)
    factor = cpd.to_factor()
    print(factor)

# check some function used for KL computation
for node in nodes:
    print("conditional for: ", node)
    conditional = model.get_cpds(node)
    variables = conditional.variables
    nrow = conditional.cardinality[0]
    print("   value of nrow: ", nrow)
    import numpy as np
    ncol = np.prod(conditional.cardinality[1:])
    print("   value of ncol: ", ncol)
    for j in range(ncol):
        print("     conditional.assignment([j]): ", conditional.assignment([j]))
        # print("     seleccionando 0: ", conditional.assignment([j])[0])
        # print("     seleccionando 0 + 1: ", conditional.assignment([j])[0][1:])
        # creates a dictionary selecting the assignments on parent variables
        # the first index (0) selects the list into the list returned
        # by assignment function. The second index (1:) select all the
        # pairs of values for parent variables and corresponding values
        observations=dict(conditional.assignment([j])[0][1:])
        print("     observations: ", observations)
