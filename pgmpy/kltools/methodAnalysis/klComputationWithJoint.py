# gets both models to be compared
import math

import numpy

from pgmpy.readwrite import BIFReader

############################################################
# computation of KL computing the global joint distribution
# of both models and computing the distance between them
############################################################

# gets base model
# baseModel = get_example_model("cancer")
reader = BIFReader("./nets/simple4N.bif")
baseModel = reader.get_model()

def safeLog(value):
    return math.log(value) if value > 0.0 else 0.0

# combine all factors
baseCPDs = baseModel.get_cpds()
baseFactors = []
for cpd in baseCPDs:
    factor = cpd.to_factor()
    print("factor class: ", type(factor))
    baseFactors.append(factor)

# combines all the factor
baseJoint = baseFactors[0].product(baseFactors[1], inplace=False)
for i in range(2, len(baseFactors)):
    baseJoint.product(baseFactors[i], inplace=True)

print(baseJoint)
print("scope: ", baseJoint.scope())
cardinality = baseJoint.get_cardinality(baseJoint.scope())
print("base joint cardinality: ", cardinality)
globalCardinality = numpy.prod(list(cardinality.values()))
print("globalCardinality: ", globalCardinality)

# gets alternative model
reader = BIFReader("./nets/simple4N2.bif")
alternativeModel = reader.get_model()

# combine all factors
altCPDs = alternativeModel.cpds
altFactors = []
for cpd in altCPDs:
    factor = cpd.to_factor()
    altFactors.append(factor)

# combines all the factor
altJoint = altFactors[0].product(altFactors[1], inplace=False)
for i in range(2, len(altFactors)):
    altJoint.product(altFactors[i], inplace=True)

print(altJoint)

# global loop
sum = 0
for index in range(0,globalCardinality):
    conf = baseJoint.assignment([index])[0]
    print("type of conf: ", type(conf))
    print("conf: ", conf)
    v1 = baseJoint.get_value(**dict(conf))
    v2 = altJoint.get_value(**dict(conf))
    print("v1 : ", v1 , " v2: ", v2)
    partial = v1*safeLog(v1) - v1*safeLog(v2)
    print("   sum value: ", sum, " partial: ", partial)
    sum += partial
    print("   updated sum: ", sum)

print("final div: ", sum)