from pgmpy.kltools.utilityFunctions import getFamilies, getCPDs, getCPDValues, KLDivergenceAlt1Pair
from pgmpy.kltools.variableEliminationKL import VariableEliminationKL
from pgmpy.readwrite import BIFReader
from pgmpy.utils import get_example_model

# read original model for cancer
reader = BIFReader("./nets/cancer.bif")
model = reader.get_model()

# shows information about nodes in the model
print("nodes in original model: \n")
print(model.nodes())

# get families
families = getFamilies(model)
print("Families in the original model: ")
print(families)
print("---------------------------------")

# get cpds for model
cpds = getCPDs(model)

print("original cpds: ")
for cpd in cpds:
    print("cpd with scope: ", cpd)
    print(".................................")
    print("potential to access: ")
    print(cpds[cpd])
    print(getCPDValues(cpds[cpd]))
print("---------------------------------")

# read alternative model
reader = BIFReader("./nets/cancer2.bif")
altModel = reader.get_model()

# shows nodes in alternative model
print("\n\nnodes in alternative model: \n")
print(altModel.nodes())

# get nodes and families
altFamilies = getFamilies(altModel)
print("Families in alternative model: ")
print(altFamilies)
print("---------------------------------")

# get cpds for alternative model
altCpds = getCPDs(altModel)

print("alternative cpds: ")
for cpd in altCpds:
    print("cpd with scope: ", cpd)
    print(".................................")
    print("potential to access: ")
    print(altCpds[cpd])
    print(getCPDValues(altCpds[cpd]))
print("---------------------------------")


# computes distributions using the method exploiting
# all the advantages of operations repository
(kl, engine) = KLDivergenceAlt1Pair(model, altModel)

# shows information about kl value
print("computed kl value: ", kl)

