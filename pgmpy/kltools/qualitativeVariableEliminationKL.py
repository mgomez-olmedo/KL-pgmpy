#!/usr/bin/env python3
import itertools

from collections import defaultdict, OrderedDict
from pgmpy.factors import factor_product

from pgmpy.kltools.bayesBall import BayesBall
from pgmpy.kltools.operationsRepository import OperationsRepository
from pgmpy.kltools.qualitativeBayesianModel import QualitativeBayesianModel
from pgmpy.kltools.qualitativeEliminationOrder import QualitativeWeightedMinFill, QualitativeMinNeighbors, \
    QualitativeMinWeight, QualitativeMinFill

# class for performing symbolic inference on a model
class QualitativeVariableEliminationKL():

    # class constructor
    def __init__(self, model):
        # keeps original models
        self.original_model = model

        # create a qualitative models
        self.model = QualitativeBayesianModel(model)

        # sets model variables
        self.variables = []
        self.variables.extend(self.model.nodes())

        # creates dict for cardinality
        self.cardinality = {}

        # sets working factors
        for node in self.original_model.nodes():
            cpd = self.original_model.get_cpds(node)
            self.cardinality[node] = cpd.variable_card

        # sets factors
        self.factors = defaultdict(list)
        for node in self.model.nodes():
            cpd = self.model.get_cpds(node)
            for var in cpd.scope():
                self.factors[var].append(cpd)

        # creates a Bayes Ball for determining relevant vars
        self.bayesBall = BayesBall(model)

        # obtains from factor the list of different factors
        # involved.
        distinct_factors = set()
        for node in model.nodes():
            node_factors = self.factors[node]

            # compose the set of distintc_factors
            distinct_factors = distinct_factors.union(set(node_factors))

        # convert the set into list
        distinct_factors = list(distinct_factors)

        # creates a repo with factors
        self.operations_repository = \
            OperationsRepository(sorted(distinct_factors, key=lambda factor : ''.join(factor.variables)))

        # defines a counter of matches (operations retrieved from
        # the repository)
        self.matches = 0

    # determine the relevant factors for computing posterior
    # for a node. It is a private method
    def _get_relevant_factors(self, node, irrelevant):
        # get all factors for node
        factors = self.factors[node]

        # determine which are relevant for computing posterior
        # on node
        relevant_factors = []
        for factor in factors:
            # add the factor if it does not contain irrelevant
            # variables
            if set(factor.scope()).isdisjoint(set(irrelevant)):
                relevant_factors.append((factor, None))

        # return relevant factors
        return relevant_factors

    # get working factors
    def _get_working_factors(self, irrelevant=set()):
        # make a dictionary where entries are nodes and
        # values sets of tuples (factor, None). The will
        # be a tuple for each factor where node appears
        # This operation if performed for each node in
        # self.factors (it is the same as iterating on
        # dictionary keys)
        # gets nodes relevant for getting factors
        nodes = sorted(set(self.factors) - set(irrelevant))

        # sets working factors
        working_factors = OrderedDict()
        for node in nodes:
            # get relevant factors
            relevantFactors = self._get_relevant_factors(node, irrelevant)

            # add a new entry to working_factors
            working_factors[node]=relevantFactors

        return working_factors

    # gets elimination order taking into account target vaiables
    # and irrelevant variables
    def _get_elimination_order(self, variables, irrelevant, elimination_order):
        # variables to_eliminate: tuple with names of variables to
        # remove via combination an marginalization: those in model
        # not contained in target set
        to_eliminate = sorted((set(self.variables) - set(variables) - set(irrelevant)))

        # Step 1: If elimination_order is a list, verify it's correct and return.
        if hasattr(elimination_order, "__iter__") \
                and (not isinstance(elimination_order, str)):
            # if there is any variable in elimination_order
            # contained in the target variables then raise and exception
            if any(
                var in elimination_order
                for var in set(variables)
            ):
                raise ValueError(
                    "Elimination order contains variables which are in"
                    " variables args"
                )
            else:
                return elimination_order

        # Step 2: If elimination order is None or a Markov model, return
        # a random order.
        elif (elimination_order is None):
            return to_eliminate

        # Step 3: If elimination order is a str, compute the order
        # using the specified heuristic.
        elif isinstance(elimination_order, str):
            heuristic_dict = {
                "weightedminfill": QualitativeWeightedMinFill,
                "minneighbors": QualitativeMinNeighbors,
                "minweight": QualitativeMinWeight,
                "minfill": QualitativeMinFill,
            }
            elimination_order = heuristic_dict[elimination_order.lower()](
                self.model
            ).get_elimination_order(nodes=to_eliminate)
            return elimination_order

    # private method for performing queries on a given target set
    # of variables
    def _variable_elimination(self, variables, operation, elimination_order="MinWeight",
                              joint=True, show_progress=False):

        # determine irrelevant vars with bayes ball analysis
        irrelevant = sorted(set(self.model.nodes()) -
                            set(self.bayesBall.get_relevant(variables)))

        # Step 2: Prepare data structures to run the algorithm.
        eliminated_variables = set()

        # Get working factors and elimination order
        working_factors = self._get_working_factors(irrelevant)
        elimination_order = self._get_elimination_order(
                    variables, irrelevant, elimination_order)

        # Step 3: Run variable elimination
        pbar = elimination_order
        for var in pbar:
            # Remove all the factors containing the variables which were
            # eliminated (as all the factors should be considered only once)
            factors = [
                factor
                for factor, _ in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]

            # combine factors and stores the required operations
            phi = self._combine_factors(factors, var)

            # proceed to marginalize phi
            # phi = getattr(phi, operation)([var], inplace=False)
            phi = self._execute_marginalization(phi, operation, var)

            # remove used working factors
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].append((phi, var))
            eliminated_variables.add(var)

        # Step 4: Prepare variables to be returned.
        final_distribution = []
        for node in working_factors:
            for factor, origin in working_factors[node]:
                if not set(factor.variables).intersection(eliminated_variables):
                    if factor not in final_distribution:
                        # append factor to final_distribution
                        final_distribution.append(factor)
                        # mark factor as result to avoid its removal
                        # self.operations_repository.mark_no_removable(factor);

        if joint:
            if isinstance(self.model, QualitativeBayesianModel):
                # completeFactor = factor_product(*final_distribution)
                completeFactor = self._combine_factors(final_distribution, None)

                # perform the normalization operation
                normalizedFactor = self._execute_normalization(completeFactor)

                # return the tuple with factor
                return normalizedFactor
            else:
                return factor_product(*final_distribution)
        else:
            query_var_factor = {}
            for query_var in variables:
                phi = factor_product(*final_distribution)
                query_var_factor[query_var] = phi.marginalize(
                    list(set(variables) - set([query_var])), inplace=False
                ).normalize(inplace=False)
            return query_var_factor

    # combine a list of factors two by two storing information
    # about combined factors. It is a private method
    def _combine_factors(self, factors, variable = None):
        # gets the first factor
        result = factors[0]

        # check if combination is required
        if len(factors) >= 2:
            # combine the first two potentials
            result = self._execute_combination(factors[0], factors[1], variable)

            # proceed with the rest of them
            for i in range(2, len(factors)):
                # gets the following one
                result = self._execute_combination(result, factors[i], variable)

        # return result
        return result

    # combines a single pair of factors. It is a private method
    # @param phi1 first potential
    # @param phi2 second potential
    def _execute_combination(self, phi1, phi2, variable = None):

        # check if such operation was previously done
        operation = self.operations_repository.check_combination(phi1, phi2)

        # if operation is None, perform the operation and store info
        if operation is None:
            # perform operation
            result = phi1*phi2

            # stores the operation in repo
            self.operations_repository.add_combination(phi1, phi2, result, variable)
        else:
            # operation was previously done and result can be retrieved
            # from operation
            self.matches += 1
            result = self.operations_repository.get_factor(operation.result_index)

            # stores the operation in repo as repeated
            self.operations_repository.add_combination(phi1, phi2, result, variable, repeated = True)

        # finally return result
        return result

    # perform a marginalization operation: private method
    def _execute_marginalization(self, phi, potential_operation, variable):

        # check if such operation was previously done
        operation = self.operations_repository.check_marginalization(variable, phi)

        # if operation is not performed, do it and store the result
        # and the operation
        if operation is None:
            result = getattr(phi, potential_operation)([variable])

            # stores the operation
            self.operations_repository.add_marginalization(variable, phi, result)
        else:
            # just retrieve the result from operation
            self.matches += 1
            result = self.operations_repository.get_factor(operation.result_index)

            # stores the operation as repeated
            self.operations_repository.add_marginalization(variable, phi,
                                                result, repeated = True)

        # return result
        return result

    # execute normalization operation
    def _execute_normalization(self, phi):
        # check if such operation was previously done
        operation = self.operations_repository.check_normalization(phi)

        # if operation is not performed, do it and store the result
        # and the operation
        if operation is None:
            # perform the normalization
            result = phi.normalize()

            # stores the operation
            self.operations_repository.add_normalization(phi, result)
        else:
            # just retrieve the result from operation
            self.matches += 1
            result = self.operations_repository.get_factor(operation.result_index)

            # stores the operation as repeated
            self.operations_repository.add_normalization(phi, result, repeated = True)

        # return result
        return result

    # method for making inference over two lists of diferent families
    def query_pair_families(self, baseTargets, altTargets, elimination_order="MinWeight"):
        # compose a global list of targets for queries. The list is composed
        # by tuples containing a key (concatenation of variables in families
        # once lexicographically ordered) and the corresponding family
        allTargets = []
        for target in baseTargets:
            # compose key joining variables in target once ordered
            key = "-".join(sorted(target))
            allTargets.append((key, target))
        for target in altTargets:
            key = "-".join(sorted(target))
            allTargets.append((key, target))

        # compose a list (tuples_key_target) filtering possibly repeated
        # tuples (in order to avoid the repetaed computation on the same
        # family several times)
        keys = set()
        tuples_key_target = []
        for tuple in allTargets:
            if tuple[0] not in keys:
                keys.add(tuple[0])
                tuples_key_target.append((tuple[0], tuple[1]))

        # initialize results as a dictionary
        results = {}

        # now proceed to computation of queries in order to
        # generate a plan
        for target in tuples_key_target:
            # compose the query for the list of variables stored
            # in the current tuple as target
            result = self.query(target[1], elimination_order)

            # store the result using the key for the target
            results[target[0]] = result

        # compose a list of results for each list of families
        baseResults = {}
        for target in baseTargets:
            key = "-".join(sorted(target))
            baseResults[key] = results[key]
        altResults = {}
        for target in altTargets:
            key = "-".join(sorted(target))
            altResults[key] = results[key]

        # return tuple with results for base families, results for
        # alt families and repository of operations
        return (baseResults, altResults, self.operations_repository)

    # method for performing posterior computations on families
    # of variables
    def query_families(self, targets, elimination_order="MinWeight"):
        # sets targets
        self.targets = targets

        # initialize results
        results = {} 

        # for each target, call query method
        for target in self.targets:
            result = self.query(target, elimination_order)
            key = "-".join(sorted(target))
            results[key] = result

        # return results and operation repository
        return (results, self.operations_repository)

    # main mathod of the class with the goal of performing symbolic
    # inference on a target set of variables
    def query(self, variables, elimination_order="MinWeight",
              joint=True):
        # call _variable_elimination method and return the result
        factor = self._variable_elimination(variables=variables,
                                          operation="marginalize",
                                          elimination_order=elimination_order,
                                          joint=joint)

        # check the id of the factor into factors repository
        factor_index = self.operations_repository.get_factor_index(factor)

        # return factor and its corresponding index
        return (factor, factor_index)

    # shows operation repository information
    def show_operations_repository(self):
        print(self.operations_repository.__str__())

    # gets the indexes where factors could be removed
    def get_factors_removable_time(self):
        return self.operations_repository.get_factors_removable_time()

    # gets the gropus of factors according to their removable time
    def group_factor_removable_times(self):
        # gets factor removable times
        factor_times = self.get_factors_removable_time()
        return self.operations_repository.group_removable_times(factor_times)
