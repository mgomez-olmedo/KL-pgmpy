#!/usr/bin/env python3

import itertools
from collections import defaultdict
import logging
from operator import mul
from functools import reduce

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from pgmpy.base import DAG
from pgmpy.factors.discrete import (
    TabularCPD,
    JointProbabilityDistribution,
    DiscreteFactor,
)
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.models import BayesianModel
from pgmpy.models.MarkovModel import MarkovModel
from pgmpy.kltools.qualitativeFactor import QualitativeFactor

# class for storing the info about a Bayesian model
# with qualitative factors
class QualitativeBayesianModel(DAG):

    # class constructor from a normal Bayesian model
    def __init__(self, model):
        # call to super constructor
        super(QualitativeBayesianModel, self).__init__()

        # add nodes and edges from model passed as argument
        for node in model.nodes():
            self.add_node(node)
        self.add_edges_from(model.edges())

        # intialize cpds and cardinalities
        self.qualitative_cpds = []
        self.cardinalities = defaultdict(int)

        # add qualitative versions of cpds
        if isinstance(model, BayesianModel):
            for cpd in model.cpds:
                self.cardinalities[cpd.variable] = cpd.cardinality[0]
                qualitative_cpd = QualitativeFactor(cpd.variables,
                                                cpd.cardinality)
                self.qualitative_cpds.append(qualitative_cpd)
        else:
            for cpd in model.qualitative_cpds:
                qualitative_cpd = QualitativeFactor(cpd.variables,
                                                cpd.cardinality)
                self.qualitative_cpds.append(qualitative_cpd)
            for node in model.nodes:
                self.cardinalities[node] = model.cardinalities[node]

    # adds an edge 
    def add_edge(self, u, v, **kwargs):
        if u == v:
            raise ValueError("Self loops are not allowed.")
        if u in self.nodes() and v in self.nodes() and nx.has_path(self, v, u):
            raise ValueError(
                "Loops are not allowed. Adding the edge from (%s->%s) forms a loop."
                % (u, v)
            )
        else:
            super(QualitativeBayesianModel, self).add_edge(u, v, **kwargs)

    def remove_node(self, node):
        # get affected nodes
        affected_nodes = [v for u, v in self.edges() if u == node]

        for affected_node in affected_nodes:
            node_cpd = self.get_cpds(node=affected_node)
            if node_cpd:
                node_cpd.marginalize([node])

        if self.get_cpds(node=node):
            self.remove_cpds(node)
        super(QualitativeBayesianModel, self).remove_node(node)

    def remove_nodes_from(self, nodes):
        for node in nodes:
            self.remove_node(node)

    def add_cpds(self, *cpds):
        for cpd in cpds:
            if not isinstance(cpd, (QualitativeFactor, ContinuousFactor)):
                raise ValueError("Only TabularCPD or ContinuousFactor can be added.")

            if set(cpd.scope()) - set(cpd.scope()).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.qualitative_cpds)):
                if self.qualitative_cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning(f"Replacing existing CPD for {cpd.variable}")
                    self.qualitative_cpds[prev_cpd_index] = cpd
                    break
            else:
                self.qualitative_cpds.append(cpd)

    def get_cpds(self, node=None):
        if node is not None:
            if node not in self.nodes():
                raise ValueError("Node not present in the Directed Graph")
            else:
                for cpd in self.qualitative_cpds:
                    # fist variable is main - variable
                    if cpd.variables[0] == node:
                        return cpd
        else:
            return self.qualitative_cpds

    def remove_cpds(self, *cpds):
        for cpd in cpds:
            if isinstance(cpd, str):
                cpd = self.get_cpds(cpd)
            self.qualitative_cpds.remove(cpd)

    def get_cardinality(self, node=None):

        if node:
            return self.get_cpds(node).cardinality[0]
        else:
            cardinalities = defaultdict(int)
            for cpd in self.qualitative_cpds:
                cardinalities[cpd.variable] = cpd.cardinality[0]
            return cardinalities

    def check_model(self):
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if cpd is None:
                raise ValueError(f"No CPD associated with {node}")
            elif isinstance(cpd, QualitativeFactor):
                evidence = cpd.get_evidence()
                parents = self.get_parents(node)
                if set(evidence if evidence else []) != set(parents if parents else []):
                    raise ValueError(
                        f"CPD associated with {node} doesn't have proper parents associated with it."
                    )

        return True

    def to_markov_model(self):
        moral_graph = self.moralize()
        mm = MarkovModel(moral_graph.edges())
        mm.add_nodes_from(moral_graph.nodes())
        mm.add_factors(*[cpd.to_factor() for cpd in self.qualitative_cpds])

        return mm

    def copy(self):
        model_copy = QualitativeBayesianModel(self)
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        return model_copy