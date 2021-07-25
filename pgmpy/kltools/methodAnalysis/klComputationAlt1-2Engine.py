import sys
import timeit

from pgmpy.kltools.utilityFunctions import KLDivergenceAlt1, KLDivergenceAlt1Pair
from pgmpy.readwrite import BIFReader

############################################################
# computes time for alternative method using a single engine
# for both models. There are 10 repetitions in order to get
# a goog estimation of computation time
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
    kl1 = KLDivergenceAlt1(baseModel, alternativeModel)
    print("Result with first method (2 engines): {:2.8f}".format(kl1))

# use first method
print("computation with KL for base and alternative")
exe1 = timeit.repeat(f1, number = 1, repeat = 10)
# kl1 = KLDivergenceAlt1(baseModel, alternativeModel)
# print("Result with first method: ", kl1)
print("times: ", exe1)
print("exe1 average: ", sum(exe1)/10)
print("Average execution time: {:4.4f}".format(sum(exe1)/10))

# shows statistics about repositories of operations and factors

