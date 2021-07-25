import numpy as np

from pgmpy.factors.discrete import TabularCPD, DiscreteFactor


# class for qualitative factors (without values) for making
# qualitative variable elimination
class QualitativeFactor(DiscreteFactor):

    def __init__(self, variables, cardinals):
        # set data members
        self.variables = variables
        self.cardinality = cardinals

    # gets a str with the content of the cpd
    def __str__(self):
        varnames = " ".join(map(str,self.variables))
        cardinals = self.get_cardinality(self.variables)
        cardinalValues = []
        for var in self.scope():
            cardinalValues.append(cardinals[var])

        cardinalities = " ".join(map(str, cardinalValues))
        return "phi(" + varnames + ") , cardinals: " + cardinalities

    # computes the product of variables cardinals
    def compute_cost(self):
        cardinals = self.get_cardinality(self.variables)
        return np.product(list(cardinals.values()))

    # perform marginalization operation
    def marginalize(self, variables):
        # determine variables to keep
        toKeep = list(set(self.variables) - set(variables))

        # make a new qualitativeFactor
        cardinals = list(map(lambda x: self.cardinality[self.variables.index(x)], toKeep))
        return QualitativeFactor(toKeep, cardinals)

    # perform marginalization operation
    def normalize(self):
        # keep all the vars
        toKeep = self.variables

        # make a new qualitativeFactor
        cardinals = list(map(lambda x: self.cardinality[self.variables.index(x)], toKeep))
        return QualitativeFactor(toKeep, cardinals)

    # perform combination operation
    def product(self, phi1):
        # determine variables of new potential
        newVars = self.variables.copy()
        for var in phi1.variables:
            if var not in newVars:
                newVars.append(var)

        # set cardinalities
        cardinals = []
        for var in newVars:
            if var in self.variables:
                varCardinal = self.get_cardinality([var])[var]
            else:
                varCardinal = phi1.get_cardinality([var])[var]
            cardinals.append(varCardinal)

        # makes a new qualitative factor
        return QualitativeFactor(newVars, cardinals)

    def __hash__(self):
        variable_hashes = [hash(variable) for variable in self.variables]
        sorted_var_hashes = sorted(variable_hashes)

        return hash(
            str(sorted_var_hashes)
        )

    def __mul__(self, other):
        return self.product(other)
