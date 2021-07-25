
from pgmpy.kltools.variableEliminationKL import VariableEliminationKL
from pgmpy.utils import get_example_model
from pgmpy.kltools.utilityFunctions import getFamilies

############################################################
# use of one of the basic functions for computing VariableELiminationKL
# method: query_families
############################################################

# determine relevant variables for the query
model = get_example_model('asia')

# makes a inference on VEKL for all the families
inference = VariableEliminationKL(model)

# query = inference.query(target, elimination_order="WeightedMinFill", joint=True)
# queries = inference.query_families([["asia"], ["smoke"], ["tub", "asia"],
#                                    ["lung", "smoke"], ["either", "tub", "lung"],
#                                    ["bronc", "smoke"], ["xray", "either"],
#                                    ["dysp", "either", "bronc"]])
families = getFamilies(model)
queries = inference.query_families(families)

# show operations repository at the end
inference.show_operations_repository()

# show result of queries
print("final list of result for queries: ")
print(type(queries))
for query in queries:
    print("key: ", query)
    print(queries[query])


