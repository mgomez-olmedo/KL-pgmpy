from pgmpy.utils import get_example_model
from pgmpy.kltools.factorsRepository import FactorsRepository

############################################################
# just some examples about using a repository of factors
############################################################

# read model for asia
model = get_example_model('asia')
print(model.nodes())

# creates a factor repository for the model
factors = []

for cpd in model.cpds:
    factors.append(cpd.to_factor())
repo = FactorsRepository(factors)

# shows repo info
print(repo)
