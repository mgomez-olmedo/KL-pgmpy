import sys
import timeit

from pgmpy.kltools.utilityFunctions import KLDivergenceAlt1, KLDivergenceAlt2, KLDivergenceAlt1Pair
from pgmpy.readwrite import BIFReader

############################################################
# computes time for alternative method using a single engine
# for both models. In general uses 10 repetitions in order to
# get a better estimation for time, but some nets required to
# reduce this number to one due to the long time required.
# NOTA: in complex models some times the execution fails due
# to the random nature to the access to python dictionaries
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

def f1():
    (kl, size) = KLDivergenceAlt2(baseModel, alternativeModel)
    print("Result with evidence method: {:2.8f}".format(kl), " max size: ", size)

# use first method
print("computation with evidence method")
exe1 = timeit.repeat(f1, number = 1, repeat = 1)
# kl1 = KLDivergenceAlt1(baseModel, alternativeModel)
# print("Result with first method: ", kl1)
print("times: ", exe1)
print("exe1 average: ", sum(exe1)/10)
print("Average execution time: {:4.4f}".format(sum(exe1)/10))

# shows statistics about repositories of operations and factors

