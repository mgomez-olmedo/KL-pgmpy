from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination

############################################################
# just some examples about using the original class
# for VariableElimination algorithm
############################################################

# read model for asia
model = get_example_model('asia')
print(model.nodes())

# make a inference engine
inference = VariableElimination(model)

# make a query
target = ["asia"]
query = inference.query(target, elimination_order="WeightedMinFill",
                        joint=True)
print(query)

target = ["smoke"]
query = inference.query(target, elimination_order="WeightedMinFill",
                        joint=True)
print(query)

target = ["tub", "asia"]
query = inference.query(target, elimination_order="WeightedMinFill",
                        joint=True)
print(query)

target = ["lung", "smoke"]
query = inference.query(target, elimination_order="WeightedMinFill",
                        joint=True)
print(query)

target = ["either", "tub", "lung"]
query = inference.query(target, elimination_order="WeightedMinFill",
                        joint=True)
print(query)

target = ["bronc", "smoke"]
query = inference.query(target, elimination_order="WeightedMinFill",
                        joint=True)
print(query)

target = ["xray", "either"]
query = inference.query(target, elimination_order="WeightedMinFill",
                        joint=True)
print(query)

target = ["dysp", "either", "bronc"]
query = inference.query(target, elimination_order="WeightedMinFill",
                        joint=True)
print(query)