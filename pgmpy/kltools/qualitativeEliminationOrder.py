from abc import abstractmethod
from collections import OrderedDict
from itertools import combinations
from tqdm import tqdm

import numpy as np

from pgmpy.kltools.qualitativeBayesianModel import QualitativeBayesianModel
from pgmpy.models import BayesianModel
from pgmpy.global_vars import SHOW_PROGRESS


class QualitativeBaseEliminationOrder:
    """
    Base class for finding elimination orders.
    """

    def __init__(self, model):
        """
        Init method for the base class of Elimination Orders.

        Parameters
        ----------
        model: BayesianModel instance
            The model on which we want to compute the elimination orders.
        """
        if not isinstance(model, QualitativeBayesianModel):
            raise ValueError("Model should be a BayesianModel instance")
        self.model = model.copy()
        self.moralized_model = self.model.moralize()

    @abstractmethod
    def cost(self, node):
        return 0

    def get_elimination_order(self, nodes=None, show_progress=True):
        if nodes is None:
            nodes = self.model.nodes()
        #nodes = set(nodes)

        ordering = []
        while nodes:
            #scores = {node: self.cost(node) for node in nodes}
            scores = OrderedDict()
            for node in nodes:
                scores[node] = self.cost(node)
            min_score_node = min(scores, key=scores.get)
            ordering.append(min_score_node)
            nodes.remove(min_score_node)
            self.model.remove_node(min_score_node)
            self.moralized_model.remove_node(min_score_node)

        return ordering

    def fill_in_edges(self, node):
        return combinations(self.model.neighbors(node), 2)


class QualitativeWeightedMinFill(QualitativeBaseEliminationOrder):
    def cost(self, node):
        edges = combinations(self.moralized_model.neighbors(node), 2)
        return sum(
            [
                self.model.get_cardinality(edge[0])
                * self.model.get_cardinality(edge[1])
                for edge in edges
            ]
        )


class QualitativeMinNeighbors(QualitativeBaseEliminationOrder):
    def cost(self, node):
        return len(list(self.moralized_model.neighbors(node)))


class QualitativeMinWeight(QualitativeBaseEliminationOrder):
    def cost(self, node):
        """
        The cost of a eliminating a node is the product of weights, domain cardinality,
        of its neighbors.
        """
        return np.prod(
            [
                self.model.get_cardinality(neig_node)
                for neig_node in self.moralized_model.neighbors(node)
            ]
        )


class QualitativeMinFill(QualitativeBaseEliminationOrder):
    def cost(self, node):
        """
        The cost of a eliminating a node is the number of edges that need to be added
        (fill in edges) to the graph due to its elimination
        """
        return len(list(self.fill_in_edges(node)))
