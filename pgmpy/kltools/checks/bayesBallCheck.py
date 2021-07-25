from pgmpy.kltools.bayesBall import *

from pgmpy.utils import get_example_model

############################################################
# just checks the functions of BayesBall class on cancer
# network and some examples of queries
############################################################

# read model for asia
model = get_example_model('cancer')
print(model.nodes())

# gets relevant variables for either and dysp
baller = BayesBall(model)

# call getRelevant method without evidence
relevant = baller.get_relevant(["Pollution", "Xray"])

# must be asia, tub, smoke, lung, bronc, either, dysp
print("relevant vars: ", relevant)

# shows the final state of the nodes
print(baller.__str__())

print("-------------------------------------------------------")

# call getRelevant passing either as evidence
# relevant = baller.get_relevant(["dysp"], ["either"])
# print("relevant vars: ", relevant)

# shows the final state of the nodes
# print(baller.__str__())
