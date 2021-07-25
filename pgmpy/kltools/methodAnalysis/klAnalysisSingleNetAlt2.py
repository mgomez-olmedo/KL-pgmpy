import sys
import timeit

from pgmpy.kltools.utilityFunctions import KLDivergenceAlt1, KLDivergenceAlt2, KLDivergenceAlt1Pair
from pgmpy.readwrite import BIFReader
from pgmpy.utils import get_example_model

############################################################
# analyzes the behaviour the method for computing KL
# considering number of operations required, number of
# repetitions for combination, marginalization and
# normalization operations, maximum size of factors required
# during the computation (if no removal of factors where
# applied) and real memory size required for storing
# potentials
# this is performed for a single network
############################################################

# just compares both KL computation methods
name = sys.argv[1]
baseName = "./nets/" + name + ".bif"
altName = "./nets/" + name + "2.bif"

# gets both models to be compared
# baseModel = get_example_model("cancer")
reader = BIFReader(baseName)
baseModel = reader.get_model()

reader = BIFReader(altName)
alternativeModel = reader.get_model()

# computes the divergence with the method of interest
(kl1, maxSize) = KLDivergenceAlt2(baseModel, alternativeModel)
print("Result with evidence method: ", kl1)

# shows statistics about max size
print("max size: ", maxSize)

# just produce the computation with the method based on using
# evidence
# result2 = KLDivergenceAlt1Pair(baseModel, alternativeModel)
# print("Result with second method: ", result2[0])

