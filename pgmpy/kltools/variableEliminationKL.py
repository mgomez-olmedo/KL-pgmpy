#!/usr/bin/env python3
import itertools

import networkx as nx
from tqdm import tqdm

from pgmpy.factors import factor_product
from pgmpy.inference import Inference
from pgmpy.inference.EliminationOrder import (
    WeightedMinFill,
    MinNeighbors,
    MinFill,
    MinWeight,
)
from pgmpy.kltools.bayesBall import BayesBall
from pgmpy.kltools.qualitativeVariableEliminationKL import QualitativeVariableEliminationKL
from pgmpy.models import BayesianModel
from pgmpy.kltools.operationsRepository import *
from pgmpy.global_vars import SHOW_PROGRESS
from pgmpy.kltools.utilityFunctions import computeSize

# class for computing joint distributions in order to
# obtain Kullback-Leibler distances. The class inherits
# from Inference
class VariableEliminationKL(Inference):

    # class constructor
    # @param model model to use for inference tasks
    def __init__(self, model):
        # call super constructor
        super().__init__(model)

        # creates an object for making the qualitative propagation
        # for detecting the operation plan and the moments where
        # potentials can be removed
        self.qualitativeEvaluator = QualitativeVariableEliminationKL(model)

        # creates a Bayes Ball for determining relevant vars
        self.bayesBall = BayesBall(model)

        # obtains from factors the list of different factors
        # involved. This is required because cpds in model
        # are transformed into factors in Inference class and
        # therefore ids of objects change. Makes as well a dictionary
        # for storing cardinalities of variables
        self.cardinalities = {}
        distinct_factors = set()
        for node in model.nodes():
            # gets the factors for node
            node_factors = self.factors[node]

            # gets the variable cardinality from one of the
            # factors
            # get_cardinality gets dictionary with entries with (node, cardinality)
            self.cardinalities[node] = node_factors[0].get_cardinality([node])[node]

            # compose the set of distintc_factors
            distinct_factors = distinct_factors.union(set(node_factors))

        # convert the set into a list
        distinct_factors = list(distinct_factors)

        # list with values of potentials produced when the operation
        # is performed without using cache
        self.sizes = []

        # creates a repo with factors
        self.operations_repository = \
            OperationsRepository(sorted(distinct_factors, key=lambda factor: ''.join(factor.variables)))

        # stores info about matches
        self.combination_matches = 0
        self.marginalization_matches = 0
        self.normalization_matches = 0
        self.number_operations = 0
        self.removed = 0

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
    def _get_working_factors(self, evidence, irrelevant=set()):
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
            working_factors[node] = relevantFactors

        # Dealing with evidence. Reducing factors over it before VE is run.
        # When there is evidence, then tuples show in its second component
        # the evidence var and factors are modified for capturing evidence
        # information
        nf = 1.0
        if evidence:
            for evidence_var in evidence:
                for factor, origin in working_factors[evidence_var]:
                    factor_reduced = factor.reduce(
                        [(evidence_var, evidence[evidence_var])], inplace=False
                    )
                    if not factor_reduced.scope():
                        nf *= factor_reduced.values

                    for var in factor_reduced.scope():
                        working_factors[var].remove((factor, origin))
                        working_factors[var].append((factor_reduced, evidence_var))
                del working_factors[evidence_var]
        return (working_factors, nf)

    # gets elimination order taking into account target vaiables
    # and irrelevant variables
    def _get_elimination_order(self, variables, irrelevant, evidence,
                               elimination_order, show_progress=False):
        # variables to_eliminate: tuple with names of variables to
        # remove via combination an marginalization: those in model
        # not contained in target set and not adding evidence
        to_eliminate = sorted((set(self.variables) - set(variables) - set(irrelevant)
                               - set(evidence.keys() if evidence else [])))

        # Step 1: If elimination_order is a list, verify it's correct and return.
        if hasattr(elimination_order, "__iter__") \
                and (not isinstance(elimination_order, str)):
            # if there is any variable in elimination_order
            # contained in the target variables or in the
            # evidence set, then raise and exception
            if any(
                    var in elimination_order
                    for var in set(variables).union(
                        set(evidence.keys() if evidence else [])
                    )
            ):
                raise ValueError(
                    "Elimination order contains variables which are in"
                    " variables or evidence args"
                )
            else:
                return elimination_order

        # Step 2: If elimination order is None or a Markov model, return
        # a random order.
        elif (elimination_order is None) or \
                (not isinstance(self.model, BayesianModel)):
            return to_eliminate

        # Step 3: If elimination order is a str, compute the order
        # using the specified heuristic.
        elif isinstance(elimination_order, str) \
                and isinstance(self.model, BayesianModel):
            heuristic_dict = {
                "weightedminfill": WeightedMinFill,
                "minneighbors": MinNeighbors,
                "minweight": MinWeight,
                "minfill": MinFill,
            }
            elimination_order = heuristic_dict[elimination_order.lower()](
                self.model
            ).get_elimination_order(nodes=to_eliminate, show_progress=show_progress)
            return elimination_order

    # method for executing a previously computed plan of operations
    # @param plan object of class OperationsRepository containing the
    # operations to perform
    def execute_plan(self, plan):
        # stores the global number of operations
        self.number_operations = plan.get_size()

        # considers each operation in the plan
        for index in range(0, self.number_operations):
            # gets the operation
            operation = plan.get_operation(index)

            # check if it is repetaed
            if operation.repeated == True:
                # add the operation as repeated
                self.operations_repository.add_operation(operation)

                # add 1 to the corresponding counter of matches
                if operation.code == OperationCode.COMBINATION:
                    self.combination_matches += 1
                elif operation.code == OperationCode.MARGINALIZATION:
                    self.marginalization_matches += 1
                else:
                    self.normalization_matches += 1

            else:
                # check kind of operation
                if operation.code == OperationCode.COMBINATION:
                    self._execute_combination(operation)
                elif operation.code == OperationCode.MARGINALIZATION:
                    self._execute_marginalization(operation)
                else:
                    self._execute_normalization(operation)

                # determine removable factor checking the plan
                removable = plan.get_removable_factors(index)

                # remove the factors on the current factors repository
                self.operations_repository.factors_repository.remove_factors(removable)

        # set the number of removed factors
        self.removed = self.operations_repository.factors_repository.removed

    # method for computing posterior for a given set of variables
    def _variable_elimination(
            self,
            variables,
            operation,
            evidence=None,
            elimination_order="MinFill",
            joint=True,
            # changed to False
            show_progress=False,
    ):
        # Step 1: Deal with the input arguments.
        if isinstance(variables, str):
            raise TypeError("variables must be a list of strings")
        if isinstance(evidence, str):
            raise TypeError("evidence must be a list of strings")

        # Dealing with the case when variables is not provided.
        if not variables:
            all_factors = []
            for factor_li in self.factors.values():
                all_factors.extend(factor_li)
            if joint:
                return factor_product(*set(all_factors))
            else:
                return set(all_factors)

        # determine irrelevant vars with bayes ball analysis
        irrelevant = sorted(set(self.model.nodes()) - set(self.bayesBall.get_relevant(variables)))

        # Step 2: Prepare data structures to run the algorithm.
        eliminated_variables = set()
        # Get working factors and elimination order
        working_factors = self._get_working_factors(evidence, irrelevant)[0]

        elimination_order = self._get_elimination_order(
            variables, irrelevant, evidence, elimination_order, show_progress=show_progress
        )

        # Step 3: Run variable elimination
        if show_progress and SHOW_PROGRESS:
            pbar = tqdm(elimination_order)
        else:
            pbar = elimination_order

        for var in pbar:
            if show_progress and SHOW_PROGRESS:
                pbar.set_description(f"Eliminating: {var}")
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [
                factor
                for factor, _ in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]

            # phi = factor_product(*factors)
            phi = self._combine_factors(factors, var)

            # proceed to marginalize phi
            # phi = getattr(phi, operation)([var], inplace=False)
            phi = self._execute_checking_marginalization(phi, operation, var)

            # remove used factors
            del working_factors[var]

            # takes into consideration the result of marginalization
            for variable in phi.variables:
                working_factors[variable].append((phi, var))
            eliminated_variables.add(var)

        # Step 4: Prepare variables to be returned.
        final_distribution = []
        for node in working_factors:
            for factor, origin in working_factors[node]:
                if not set(factor.variables).intersection(eliminated_variables):
                    if factor not in final_distribution:
                        final_distribution.append(factor)
        # selects factors in tuples avoiding repeates values
        # final_distribution = [factor for factor, _ in final_distribution]

        if joint:
            if isinstance(self.model, BayesianModel):
                # completeFactor = factor_product(*final_distribution)
                completeFactor = self._combine_factors(final_distribution, None)

                # perform the normalization operation
                normalizedFactor = self._execute_checking_normalization(completeFactor)

                # return normalized factor
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

    # method for computing posteriors without normnalization
    # and without cache of operations
    def _variable_elimination_no_cache(self, variables, operation,
                                       evidence=None,
                                       elimination_order=None):
        # Step 1: Deal with the input arguments.
        if isinstance(variables, str):
            raise TypeError("variables must be a list of strings")
        if isinstance(evidence, str):
            raise TypeError("evidence must be a list of strings")

        # Dealing with the case when variables is not provided.
        if not variables:
            all_factors = []
            for factor_li in self.factors.values():
                all_factors.extend(factor_li)

            # return all the factors
            return set(all_factors)

        # Step 2: Prepare data structures to run the algorithm.
        eliminated_variables = set()
        # Get working factors
        results = self._get_working_factors(evidence)

        working_factors = results[0]
        nf = results[1]

        # gets elimination order
        if not elimination_order:
            elimination_order = list(set(self.variables) -
                                     set(variables) -
                                     set(evidence.keys() if evidence else []))
        elif any(var in elimination_order for var in
                 set(variables).union(set(evidence.keys() if evidence else []))):
            raise ValueError("Elimination order contains variables in variables or evidence")

        # removes variables given by elimination order
        for var in elimination_order:
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            # Composes a set of variables for determining the global size of
            # the potential to produce by combination
            variablesSet = set()
            for factor in working_factors[var]:
                variablesSet = variablesSet.union(set(factor[0].scope()))

            # computes the global size to obtain from the combination
            self.sizes.append(computeSize(variablesSet, self.cardinalities))

            factors = [
                factor
                for factor, _ in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]

            # multiply factors
            phi = factor_product(*factors)
            phi = getattr(phi, operation)([var], inplace=False)
            if not phi.variables:
                nf *= phi.values

            # remove working factors of var
            del working_factors[var]

            # adds new factors
            for variable in phi.variables:
                working_factors[variable].append((phi, var))

            # add removed var to eliminated_variables
            eliminated_variables.add(var)

        # considers the final distribution
        final_distribution = set()
        for node in working_factors:
            factors = working_factors[node]
            for factor in factors:
                factor = factor[0]
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.add(factor)

        # considers now query factors
        query_var_factor = {}
        for query_var in variables:
            phi = factor_product(*final_distribution)
            query_var_factor[query_var] = phi.marginalize(
                list(set(variables) - set([query_var])),
                inplace=False) * nf

        # return query_var_factor
        return query_var_factor

    # combine a list of factors two by two storing information
    # about combined factors. It is a private method
    def _combine_factors(self, factors, variable=None):
        # gets the first factor
        result = factors[0]

        # check if combination is required
        if len(factors) >= 2:
            # combine the first two potentials
            result = self._execute_checking_combination(factors[0], factors[1], variable)

            # proceed with the rest of them
            for i in range(2, len(factors)):
                # gets the following one
                result = self._execute_checking_combination(result, factors[i], variable)

        # return result
        return result

    # combines a single pair of factors. It is a private method
    def _execute_checking_combination(self, phi1, phi2, variable=None):
        # check if such operation was previously done
        operation = self.operations_repository.check_combination(phi1, phi2)

        # if operation is None, perform the operation and store info
        if operation is None:
            # perform operation
            result = phi1 * phi2

            # stores the operation in repo
            self.operations_repository.add_combination(phi1, phi2, result, variable)
        else:
            # operation was previously done and result can be retrieved
            # from operation
            self.combination_matches += 1
            result = self.operations_repository.get_factor(operation.result_index)

            # stores operation in repo as repeated
            self.operations_repository.add_combination(phi1, phi2, result, variable, repeated=True)

        # finally return result
        return result

    # execute a planned operation of combination. This method os called
    # if the operation is not repeated
    def _execute_combination(self, operation):
        # gets ids of involved factors
        phi1 = self.operations_repository.get_factor(operation.phi1_index)
        phi2 = self.operations_repository.get_factor(operation.phi2_index)

        # perform the target operation
        result = phi1 * phi2

        # stores the operation in repo
        self.operations_repository.add_combination(phi1, phi2, result, operation.variable)

    # perform a marginalization operation: private method
    def _execute_checking_marginalization(self, phi, potential_operation, variable):

        # check if such operation was previously done
        operation = self.operations_repository.check_marginalization(variable, phi)

        # if operation is not performed, do it and store the result
        # and the operation
        if operation is None:
            result = getattr(phi, potential_operation)([variable], inplace=False)

            # stores the operation
            self.operations_repository.add_marginalization(variable, phi, result)
        else:
            # just retrieve the result from operation
            self.marginalization_matches += 1
            result = self.operations_repository.get_factor(operation.result_index)

            # adds the operation as repeated
            self.operations_repository.add_marginalization(variable, phi, result, repeated=True)

        # return result
        return result

    # execute a planned operation of marginalization. This operation is called
    # for non repeated operations
    def _execute_marginalization(self, operation):
        # gets ids of involved factors
        phi = self.operations_repository.get_factor(operation.phi1_index)

        # perform the target operation
        result = getattr(phi, "marginalize")([operation.variable], inplace=False)

        # stores the operation in repo
        self.operations_repository.add_marginalization(operation.variable, phi, result)

    # execute normalization operation
    def _execute_checking_normalization(self, phi):
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
            self.normalization_matches += 1
            result = self.operations_repository.get_factor(operation.result_index)

            # stores the operation as repeated
            self.operations_repository.add_normalization(phi, result, repeated = True)

        # return result
        return result

    # execute a planned operation of normalization. This operation is called
    # if the operation is not repeated
    def _execute_normalization(self, operation):
        # gets the factor to normalize
        phi = self.operations_repository.get_factor(operation.phi1_index)

        # perform operation
        result = phi.normalize(inplace=False)

        # stores the operation
        self.operations_repository.add_normalization(phi, result)

    # method for computing the posterior of two sets of targets
    def query_pair_families(self, baseTargets, altTargets, elimination_order="MinWeight"):
        # gets evaluation plan produced by the symbolic propagation
        # using the qualitative evaluator
        (baseResults, altResults, plan) = \
            self.qualitativeEvaluator.query_pair_families(baseTargets,
                                                          altTargets, elimination_order)

        # print("----------------- targets -------------------------------------\n")
        # print(baseTargets)
        # print("---------------------------------------------------------------\n");
        # print(altTargets)
        # print("------------------ operations plan-----------------------------\n")
        # print(plan)
        # print("---------------------------------------------------------------\n")

        # execute the plan
        self.execute_plan(plan)

        # compose results for base targets
        base_results = {}
        for target in baseTargets:
            # compose the key joining the variables in target but ordered
            key = "-".join(sorted(target))

            # gets the result for the target
            data = baseResults[key]

            # gets factor index
            factor_index = data[1]

            # gets factor from repository
            factor = self.operations_repository.get_factor(factor_index)

            # and entry to base_results
            base_results[key] = factor

        # compose results for alt families
        alt_results = {}
        for target in altTargets:
            key = "-".join(sorted(target))

            # gets the result for the target
            data = altResults[key]

            # gets factor index
            factor_index = data[1]

            # gets factor from repository
            factor = self.operations_repository.get_factor(factor_index)

            # and entry to base_results
            alt_results[key] = factor

        # return the tuple with all the results
        return (base_results, alt_results)

    # method for performing posterior computations on families
    # of variables
    def query_families(self, targets, elimination_order="MinWeight"):
        # sets targets
        self.targets = targets

        # gets planification about factors removal using the same elimination
        # order. The method of evaluation returns a tuple with results and
        # the repository of operations. Results is a dictionary containing the
        # key of the target (sorted list of variables joined by -)
        (results, plan) = self.qualitativeEvaluator.query_families(targets, elimination_order)

        # just execute the plan
        self.execute_plan(plan)

        # update results with potentials obtained from qualitative evaluation
        final_results = {}
        for key in results.keys():
            data = results[key]
            factor_index = data[1]
            factor = self.operations_repository.get_factor(factor_index)
            # add entry to final results
            final_results[key] = factor

        # initialize results
        # results = {}

        # for each target, call query method
        # for target in self.targets:
        #   result = self.query(target, elimination_order = elimination_order)
        #   key = "-".join(sorted(target))
        #   results[key] = result

        # return results
        return final_results

    # gets induced graph
    def induced_graph(self, elimination_order):
        # If the elimination order does not contain the same variables
        # as the model
        if set(elimination_order) != set(self.variables):
            raise ValueError(
                "Set of variables in elimination order"
                " different from variables in model"
            )

        eliminated_variables = set()
        working_factors = {
            node: [factor.scope() for factor in self.factors[node]]
            for node in self.factors
        }

        # The set of cliques that should be in the induced graph
        cliques = set()
        for factors in working_factors.values():
            for factor in factors:
                cliques.add(tuple(factor))

        # Removing all the factors containing the variables which are
        # eliminated (as all the factors should be considered only once)
        for var in elimination_order:
            factors = [
                factor
                for factor in working_factors[var]
                if not set(factor).intersection(eliminated_variables)
            ]
            phi = set(itertools.chain(*factors)).difference({var})
            cliques.add(tuple(phi))
            del working_factors[var]
            for variable in phi:
                working_factors[variable].append(list(phi))
            eliminated_variables.add(var)

        edges_comb = [
            itertools.combinations(c, 2) for c in filter(lambda x: len(x) > 1, cliques)
        ]
        return nx.Graph(itertools.chain(*edges_comb))

    # gets induced width
    def induced_width(self, elimination_order):
        """
        Returns the width (integer) of the induced graph formed by running Variable Elimination on the network.
        The width is the defined as the number of nodes in the largest clique in the graph minus 1.

        Parameters
        ----------
        elimination_order: list, array like
            List of variables in the order in which they are to be eliminated.

        Examples
        --------
        """
        induced_graph = self.induced_graph(elimination_order)
        return nx.graph_clique_number(induced_graph) - 1

    # shows operation repository information
    def show_operations_repository(self):
        print(self.operations_repository.__str__())

    # return the max size stored in sizes
    def getMaxSize(self):
        return max(self.sizes)
