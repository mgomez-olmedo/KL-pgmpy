# class for storing the factors defining a model and
# produced during model evaluation
import gc

import numpy as np


# class for storing factors involved in variable elimination
# operations
class FactorsRepository:
    # class constructor. The argument contains the initial
    # list of factors
    def __init__(self, factors):
        self.factors = []
        # for each factor store a tupla with factor-id and
        # factor itself
        for factor in factors:
            # for each factor a tuple with the following information
            # is stored: (id of factor, factor, removable flag,
            # las operation where the factor is used, size). Initially
            # original factors as marked as non removable and all
            # of them are used in 0 operation. This last part will
            # change when the factor is used in an operation. The cost
            # stores the number of values of the factor
            self.factors.append((id(factor), factor, False, 0, 0))

        # compute cost for the whole repository
        (self.real_cost, self.max_cost) = self.compute_cost()

        # defines a threshold for calling garbage collector
        self.threshold = 1000000

        # counter of removed factors: will be used as id for removed
        # factors
        self.removed = 0

    # add a new factor if not present. It assumes a previous
    # check is done before adding the factor
    # return the index where the the factor was stored
    def add(self, factor, removable=True):
        # check if the factor was previously stored
        result = self.get_factor_index(factor)

        # just append the factor to the repo if needed and set
        # it as removable. Initially information about last operation
        # is set to 0 (afterwards it will be changed with update_operation
        # method)
        if result == -1 or (result != -1 and self.factors[result][1] == None):
            self.factors.append((id(factor), factor, removable, 0, self._compute_factor_cost(factor)))
            result = len(self.factors) - 1

        # return result
        return result

    # gets the position of a certain factor
    def get_factor_index(self, factor):
        result = -1
        for i in range(0, len(self.factors)):
            # check the id part of the tuple (id, factor)
            # if self.factors[i][0] == id(factor) and self.factors[i][1] != None:
            if self.factors[i][0] == id(factor):
                result = i
                return result

        # return result -1 if not found
        return result

    # get the factor stored in a certain position
    def get_factor(self, index):
        data = None
        if index >= 0 and index < self.size():
            data = self.factors[index][1]

        # return data
        return data

    # gets the complete data about a factor
    def get_factor_data(self, index):
        # return the complete tuple with factor data: id, factor and
        # removable flag
        return self.factors[index]

    # return the size of the repository
    def size(self):
        return len(self.factors)

    # compute the cost for all the factors
    def compute_cost(self):
        size = 0
        max = 0

        # computes the cost for each factor
        for tuple in self.factors:
            if tuple[1] != None:
                size += tuple[4]
            # add it to max anyway
            max += tuple[4]

        # assign cost values: real and max
        (self.real_cost, self.max_cost) = (size, max)

        # return size
        return (size, max)

    # computes the size of a factor
    def _compute_factor_cost(self, factor):
        size = 0
        if factor != None:
            factor_card = factor.get_cardinality(factor.scope())
            size += np.product(list(factor_card.values()))

        # return size
        return size

    # mark as no removable the factor passed as argument
    def mark_no_removable(self, factor):
        factor_index = self.get_factor_index(factor)

        # gets the entry
        entry = self.factors[factor_index]

        # compose a new entry
        new_entry = (entry[0], entry[1], False, entry[3])

        # update entry in the list
        self.factors[factor_index] = new_entry

    # updates the index of the last operation where the factor
    # was used
    def update_operation(self, factor_index, operation_index):
        # gets tuple with factor data
        entry = self.factors[factor_index]

        # compose a new entry for updating the operation index
        new_entry = (entry[0], entry[1], entry[2], operation_index, entry[4])

        # update the entry
        self.factors[factor_index] = new_entry

    # gets the list of removable factors one this operation is performed
    def get_removable_factors(self, operation_index):
        # initializes an empty list of removable factors
        removable = []

        # considers factors one by one and add the corresponding index
        # if it is removable one operation_index is performed
        for factor_index in range(0, len(self.factors)):
            # gets information about factor in factor_index position
            entry = self.factors[factor_index]

            # it must be removable and its time corresponds to operation_index
            # (equals to avoid removal of previously removed factors)
            if entry[2] == True and entry[3] == operation_index:
                removable.append(factor_index)

        # return the list of removable factors
        return removable

    # remove the factors passed as argument as a list
    # the list contains a sequence of indices
    def remove_factors(self, factor_indices):
        # remove factors in factors_list
        for factor_index in factor_indices:
            stored_factor = self.factors[factor_index]

            # remove the factor
            self.remove_factor(factor_index)

        # recompute cost
        self.compute_cost()

    # remove the factor from the repository freeing its memory
    def remove_factor(self, factor_index):
        # gets factor data
        stored_factor = self.factors[factor_index]

        # compute size of factor
        cost = self._compute_factor_cost(stored_factor[1])

        # replace factor data
        self.factors[factor_index] = (stored_factor[0], None, stored_factor[2],
                                      stored_factor[3], stored_factor[4])
        self.removed = self.removed+1

        # delete tuple object with its content
        del stored_factor

        # call gc if cost is above threshold
        if cost >= self.threshold:
            gc.collect()

    # get the number of factor removals
    def get_removed(self):
        return self.removed

    # remove those factors with removal time equals to the
    # operation index passed as argument
    def remove_with_time(self, op_index):
        # consider all factors
        for factor_index in range(0, len(self.factors)):
            stored_factor = self.factors[factor_index]

            # remove factor values if factor is removable and
            # its time matches op_index
            if stored_factor[2] == True and stored_factor[3] == op_index:
                self.remove_factor(factor_index)

    # produces a string with factors info
    def __str__(self):
        result = "\n---------------- repo of factors--------------\n"
        index = 0
        for tuple in self.factors:
            result += "id: " + str(tuple[0]) + " index: " + str(index) + " removable: " + str(
                tuple[2]) + " last op.: " + str(tuple[3]) + "\n"
            index += 1
            factor = tuple[1]
            if factor != None:
                # result += factor.__str__() + "\n"
                result += "scope: "
                result += " ".join(factor.scope()) + "\n"
                result += ".........................................\n"
            else:
                result += "Removed\n"
                result += ".........................................\n"
        result += "real cost: " + str(self.real_cost) + "\n"
        result += "max cost: " + str(self.max_cost) + "\n"
        result += "---------------------------------------------\n"
        return result
