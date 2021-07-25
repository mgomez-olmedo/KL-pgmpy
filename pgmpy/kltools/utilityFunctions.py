
# utility functions

# gets the families for a givel model passed as argument
import math

import numpy
import numpy as np
from collections import Iterable

############################################################
# utility methods for Kullback-Leibler computation
############################################################

# computes KL divergence between two models defined of the
# same set of variables
# @param base base model
# @param alternative model to compare with respect base

# compute KL divergence with alternative method based on
# using inference
# @param base list of families in reference network
# @param alternative list of families in alternative model
# @return kl value
def KLDivergenceAlt2(base, alternative):
    (baseBase, mi1) = expLL(base, base)
    (baseAlternative, mi2) = expLL(base, alternative)
    # gets max sizes from mi1 and mi2
    maxSizes = []
    maxSizes.append(mi1.getMaxSize())
    maxSizes.append(mi2.getMaxSize())

    # use expLL for base and alternative and return the difference
    # and the global max
    return (baseBase - baseAlternative, max(maxSizes))

# method for computing log likelihood for a model
def expLL(base, alternative):
    from pgmpy.kltools.variableEliminationKL import VariableEliminationKL

    # initializes sum
    sum = 0.0

    # consider nodes in model
    for n in alternative.nodes():
        # get conditional for n
        conditional = alternative.get_cpds(n)

        # creates a engine for base
        mi = VariableEliminationKL(base)

        # get variables in conditional
        variables = conditional.variables

        # get the number of states for variable and the
        # number of configurations for parent variables
        nrow = conditional.cardinality[0]
        ncol = np.prod(conditional.cardinality[1:])

        # considers each parent configuration
        for j in range(ncol):
            # get observations: gets the assigment for all variables
            # in the conditional for n node and consistent with j-th
            # index. El value of n-variable will be the key and and
            # the rest of the assignments will be the value
            observations = dict(conditional.assignment([j])[0][1:])
            if len(observations) == 0:
                observations = None

            # compute posterior for all the observations
            res = mi._variable_elimination_no_cache([variables[0]],
                                                    "marginalize",
                                                    observations)[variables[0]]

            # consider states for n
            for i in range(nrow):
                if conditional.get_values()[i][j] > 0:
                    sum += res.values[i]*math.log(conditional.get_values()[i][j])

    # return a tuple with sum and mi
    return (sum, mi)


# method of KL computation using inference on families of
# variables
# @param base list of families in reference network
# @param alternative list of families in alternative model
# @return kl value
# NOTE: this method uses a different engine for computing
# the posteriors of each list of families. Therefore, the
# advantages of using the operations repository is not properly
# exploted
def KLDivergenceAlt1(base, alternative):
    from pgmpy.kltools.variableEliminationKL import VariableEliminationKL

    # get cpds for base and alternative model
    baseCpds = getCPDs(base)
    altCpds = getCPDs(alternative)

    # get nodes and parents for base and alternative
    baseFamilies = getFamilies(base)
    altFamilies = getFamilies(alternative)

    # creates an inference engine for base model
    engine = VariableEliminationKL(base.copy())

    # get joint distributions for base families
    joints = engine.query_families(baseFamilies)

    # get posterior for families defined in alternative model
    engine = VariableEliminationKL(base.copy())
    posteriors = engine.query_families(altFamilies)

    # computes logLike of joints and baseCpds
    pp = netLoglike(joints, baseCpds)

    # computes logLike of posteriors and altCpds
    pq = netLoglike(posteriors, altCpds)

    # compose the final value
    kl = pp - pq

    # return kl value
    return kl

# method of KL computation using inference on families of
# variables
# @param base list of families in reference network
# @param alternative list of families in alternative model
# @return tuple with kl value and the inference engine
# used for computation. The engine allows to access statistics
# about the computation process
def KLDivergenceAlt1Pair(base, alternative):
    from pgmpy.kltools.variableEliminationKL import VariableEliminationKL

    # get cpds for base and alternative model
    baseCpds = getCPDs(base)
    altCpds = getCPDs(alternative)

    # get nodes and parents for base and alternative
    baseFamilies = getFamilies(base)
    altFamilies = getFamilies(alternative)

    # creates an inference engine for base model
    engine = VariableEliminationKL(base.copy())

    # get joint distributions for base and alternative families
    (joints, posteriors) = engine.query_pair_families(baseFamilies, altFamilies)

    # compute the value comparing joints and baseCpds
    pp = netLoglike(joints, baseCpds)
    # print("end of pp computation: pp value = ", pp)

    # compute the value comparing posteriors and altCpds
    pq = netLoglike(posteriors, altCpds)
    # print("end of pq computation: pq value = ", pq)
    # print("kl computation with pp: ", pp, " pq: ", pq, " kl: ", (pp-pq))

    # compose the final value
    kl = pp - pq

    # return kl value and engine for getting statistics information
    # about the use of the repositories
    return (kl, engine)

# get all the cpds for a model
# return a dictionary of entries (scope - factor)
# the scope contains cpd variables sorted in lexicographic order
# @param model target model
# @return dictionary of entries (string with concatenation of
# domain variables, cpd)
# NOTE: the variables in the domain are lexicographicaly ordered
# before composing the key
def getCPDs(model):
    cpds = {}
    for node in model.nodes():
       cpd = model.get_cpds(node)
       scope = "-".join(sorted(cpd.scope()))
       cpds[scope] = cpd
    
    # return the dictionary
    return cpds

# get families for model variables
# @param model target model
# @return list of families for each variable
def getFamilies(model):
    # for each node, get parents and return a list with
    # node + parents
    return list(map(lambda node: model.get_parents(node) + [node], model.nodes()))

# gets all the values of a cpd but as a single list
# @param cpd target cpd
# @return list of values stored in the cpd
def getCPDValues(cpd):
    values = []
    for xs in cpd.values:
        if isinstance(xs, Iterable):
            values.extend(xs.flatten())
        else:
            values.append(xs)

    return values

# computes the product of joints and log of conditionals
# @param joints dictionary of joint distributions (cpds)
# @param conditionals dictionary of conditionals (cpds)
# @return computed value
def netLoglike(joints, conditionals):
    sum = 0
    for joint in joints:
        jointCpd = joints[joint]
        conditionalCpd = conditionals[joint]
        # accumulates sum for all cpds
        sum += cpdLoglike(jointCpd, conditionalCpd.to_factor())

    return sum

# computes loglike for two concrete distributions
# NOTE: must be defined on the same domains
def cpdLoglike(cpd1, cpd2):
    # gets both factors with the same order for variables
    cardinalities = cpd1.get_cardinality(cpd1.scope())
    card = numpy.prod(list(cardinalities.values()))

    # loop for values computation
    sum = 0.0
    for index in range(0, card):
        # make a configuration of values for getting cpds
        # values
        conf = cpd1.assignment([index])[0]
        v1 = cpd1.get_value(**dict(conf))
        v2 = cpd2.get_value(**dict(conf))
        sum += v1*safeLog(v2)

    # just return sum
    # print("   value: ", sum)
    return sum

# computes the log avoiding the problem of 0 values
def safeLog(value):
    return 0.0 if value == 0.0 else math.log(value)

# computes the global size related to a set of variables
# multiplying all the cardinalities
# @param variableSet set of variables
# @param cardinalities dictionary with entries (variable - cardinality)
def computeSize(variableSet, cardinalities):
    size = 1.0
    for variable in variableSet:
        size = size*cardinalities[variable]

    # return the computed size
    return size
