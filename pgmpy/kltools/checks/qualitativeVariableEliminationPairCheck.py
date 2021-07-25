import sys

from pgmpy.kltools.qualitativeVariableEliminationKL import QualitativeVariableEliminationKL
from pgmpy.readwrite import BIFReader
from pgmpy.kltools.utilityFunctions import getFamilies

############################################################
# test on using qualitative evaluation of a pair of models
# (the model is passed as an argument)
############################################################

# define the base and alternative name for the nets to analyze
name = sys.argv[1]
baseName = "./nets/" + name + ".bif"
altName = "./nets/" + name + "2.bif"

# gets both models to be compared
# baseModel = get_example_model("cancer")
reader = BIFReader(baseName)
baseModel = reader.get_model()

reader = BIFReader(altName)
alternativeModel = reader.get_model()

# make a inference engine
inference = QualitativeVariableEliminationKL(baseModel)

# gets families
baseFamilies = getFamilies(baseModel)
altFamilies = getFamilies(alternativeModel)

# perform a query
results = inference.query_pair_families(baseFamilies, altFamilies)

# results content: tuple with
# (0) dictionary for results of queries for base families
# (1) dictionary for results of alternative families
# (2) operations repository

# shows the results
print("results of queries for base families-----: ")
for result in results[0]:
    print(results[0][result])

print("results of queries for alt families: ")
for result in results[1]:
    print(results[1][result])

# shows operations repository
print("operations repository: ")
inference.show_operations_repository()
print(results[2])

print("operations ----  repeated ------- real size of factors  ---------- max size of factors\n")
print(results[2].get_size(), "  ", results[2].get_repetitions(), "   ",
      results[2].get_real_factors_size(), "    ", results[2].get_max_factors_size())

