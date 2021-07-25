import sys
import timeit

from pgmpy.kltools.utilityFunctions import KLDivergenceAlt1, KLDivergenceAlt2
from pgmpy.readwrite import BIFReader

############################################################
# compares the times for computing KL with both methods. The
# computation for the new method is performed with a different
# engine for each model. Therefore, times should be bigger
# than using a single engine
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
    print("Result with first method: {:2.8f}".format(kl1))

def f2():
    (kl2, size) = KLDivergenceAlt2(baseModel, alternativeModel)
    print("Result with second method: {:2.8f}".format(kl2))

# define the nuber of repetitions
repetitions = 10

# use first method
print("computation with KL for base and alternative")
exe1 = timeit.repeat(f1, number = 1, repeat = repetitions)
# kl1 = KLDivergenceAlt1(baseModel, alternativeModel)
# print("Result with first method: ", kl1)
print("exe1: ", exe1)
avg1 = sum(exe1)/repetitions
print("exe1 average: {:4.4f}".format(avg1))

print("starting computation with second method")
# kl2 = KLDivergenceAlt2(baseModel, alternativeModel)
exe2 = timeit.repeat(f2, number = 1, repeat = repetitions)
#print("Result with second method: ", kl2)
print("exe2: ", exe2)
avg2 = sum(exe2)/repetitions
print("exe2 average: {:4.4f}".format(avg2))

print("{:4.4f}   {:4.4f}".format(avg1, avg2))

