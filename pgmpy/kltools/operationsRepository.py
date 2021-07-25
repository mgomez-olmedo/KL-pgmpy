from collections import OrderedDict

from pgmpy.kltools.factorsRepository import FactorsRepository


# class OperationCode for labelling combination
# and marginalization operations
class OperationCode:
    COMBINATION = 1
    MARGINALIZATION = 2
    NORMALIZATION = 3


# class operation for storing both combination and
# marginalization operations
class Operation:
    # class constructor: phi2 is optional because it is not
    # required for marginalization operations
    # @param code operation code (combination or marginalization)
    # @param variable variable triggering this operation
    # @param phi1_index index of phi1 into factors repository
    # @param result_index index of result into factors repository
    # @param cost cost of potentials stored in factors repository
    # @param phi2_index index of phi2 into factors repository
    # @param repeated used for storing repeated operations
    def __init__(self, code, variable, phi1_index, result_index,
                 cost, phi2_index=-1, repeated=False):
        self.code = code
        self.variable = variable
        self.phi1_index = phi1_index
        self.phi2_index = phi2_index
        self.result_index = result_index
        self.cost = cost
        self.repeated = repeated

    # gets access to cost value
    def getCost(self):
        return self.cost;

    # gets s string with the information of an operation
    def __str__(self):
        result = "---------------- Operation ------------------------\n"
        result += "code: " + str(self.code)
        result += "       repeated: " + str(self.repeated)
        if (self.variable == None):
            result += " variable: " + "None" + "\n"
        else:
            result += " variable: " + self.variable + "\n"
        result += " id_phi1: " + str(self.phi1_index)
        result += " id_phi2: " + str(self.phi2_index)
        result += " id_result: " + str(self.result_index)
        result += " cost: " + str(self.cost)
        result += "\n---------------------------------------------------\n"

        # return result
        return result


# class for storing operations performed for computation
class OperationsRepository:
    # static counter for assigning entries to dictionary
    counter = 0

    # class constructor
    def __init__(self, factors):
        # defines the repository of factors
        self.factors_repository = FactorsRepository(factors)

        # initialize an empty repository (dictionary)
        self.repository = OrderedDict()

    # just create a new operation as a copy of the one passed as argument
    def add_operation(self, operation):
        # computes ans store cost
        factors_cost = self.factors_repository.compute_cost()

        # make a new Operation
        operation = Operation(operation.code, operation.variable,
                              operation.phi1_index, operation.result_index,
                              factors_cost, operation.phi2_index,
                              operation.repeated)

        # update factors information
        self.factors_repository.update_operation(operation.phi1_index, self.counter)
        self.factors_repository.update_operation(operation.phi2_index, self.counter)
        self.factors_repository.update_operation(operation.result_index, self.counter)

        # insert operation into repository
        self.repository[self.counter] = operation

        # add counter
        self.counter += 1

    # Adds a new combination operation. It is supposed a previous check
    # was conducted to avoid storing repeated operations
    def add_combination(self, phi1, phi2, result, variable=None, repeated=False):
        # find phi1 and phi2 indexes
        phi1_index = self.factors_repository.get_factor_index(phi1)
        phi2_index = self.factors_repository.get_factor_index(phi2)

        # order ids for storing the lower one in first place
        if phi1_index > phi2_index:
            auxiliar = phi1_index
            phi1_index = phi2_index
            phi2_index = auxiliar

        # store result into factors repository
        result_index = self.factors_repository.add(result)

        # computes ans store cost
        factors_cost = self.factors_repository.compute_cost()

        # make a new Operation
        operation = Operation(OperationCode.COMBINATION, variable, phi1_index, result_index,
                              factors_cost, phi2_index, repeated)

        # update information for involved factors about the last
        # operation when used (only for result if the operation is
        # repeated)
        if repeated == False:
            self.factors_repository.update_operation(phi1_index, self.counter)
            self.factors_repository.update_operation(phi2_index, self.counter)
        self.factors_repository.update_operation(result_index, self.counter)

        # insert operation into repository
        self.repository[self.counter] = operation

        # add counter
        self.counter += 1

    # Adds a new marginalization operation. It is supposed a previous check
    # was conducted in order to avoid repeated operations
    def add_marginalization(self, variable, phi, result, repeated=False):
        # get index of phi1
        phi_index = self.factors_repository.get_factor_index(phi)

        # store result into factors repo
        result_index = self.factors_repository.add(result)

        # computes ans store cost
        factors_cost = self.factors_repository.compute_cost()

        # make a new Operation
        operation = Operation(OperationCode.MARGINALIZATION, variable, phi_index,
                              result_index, factors_cost, phi2_index=-1,
                              repeated=repeated)

        # update information about last operation for factors (only for the
        # result if the operation is repeated)
        if repeated == False:
            self.factors_repository.update_operation(phi_index, self.counter)
        self.factors_repository.update_operation(result_index, self.counter)

        # insert operation into repository
        self.repository[self.counter] = operation

        # add counter
        self.counter += 1

    def add_normalization(self, phi, result, variable=None, repeated=False):
        # get index of phi1
        phi_index = self.factors_repository.get_factor_index(phi)

        # store result into factors repo
        result_index = self.factors_repository.add(result, removable=False)

        # computes and store cost
        factors_cost = self.factors_repository.compute_cost()

        # make a new operation
        operation = Operation(OperationCode.NORMALIZATION, variable,
                              phi_index, result_index, factors_cost,
                              phi2_index=-1, repeated=repeated)

        # update information about last operation for factors
        self.factors_repository.update_operation(phi_index, self.counter)
        self.factors_repository.update_operation(result_index, self.counter)

        # insert operation into repository
        self.repository[self.counter] = operation

        # add 1 to counter
        self.counter += 1

    # check the existence of an operation of combination on two
    # factors
    # @param phi1 first potential
    # @param phi2 second potential
    def check_combination(self, phi1, phi2):
        # gets indexes of phi1 and phi2
        phi1_index = self.factors_repository.get_factor_index(phi1)
        phi2_index = self.factors_repository.get_factor_index(phi2)

        # order them for considering the lower in first place
        if phi1_index > phi2_index:
            auxiliar = phi1_index
            phi1_index = phi2_index
            phi2_index = auxiliar

        # considers keys in operations dictionary
        for key in self.repository:
            # get entry
            entry = self.repository[key]

            # check entry values
            if entry.code == OperationCode.COMBINATION and entry.phi1_index == phi1_index \
                    and entry.phi2_index == phi2_index:
                return entry

        # if this point is reached return None
        return None

    # checks the existence of a marginalization operation
    def check_marginalization(self, variable, phi):
        # get id for phi
        id_phi = self.factors_repository.get_factor_index(phi)

        # considers dictionary entries
        for key in self.repository:
            # get entry
            entry = self.repository[key]

            # check entry values
            if entry.code == OperationCode.MARGINALIZATION and entry.phi1_index == id_phi \
                    and variable == entry.variable:
                return entry

        # if this point is reached return None
        return None

    # checks the existence of a normalization operation
    def check_normalization(self, phi):
        # get id for phi
        id_phi = self.factors_repository.get_factor_index(phi)

        # considers dictionary entries
        for key in self.repository:
            # get entry
            entry = self.repository[key]

            # check entry values
            if entry.code == OperationCode.NORMALIZATION and entry.phi1_index == id_phi:
                return entry

        # if this point is reached return None
        return None

    # mark the factor passed as argument as no removable
    def mark_no_removable(self, factor):
        # propagate the operation to factor repository
        self.factors_repository.mark_no_removable(factor)

    # gets the the factor with a given index
    def get_factor(self, index):
        return self.factors_repository.get_factor(index)

    # gets the index of a factor
    def get_factor_index(self, factor):
        return self.factors_repository.get_factor_index(factor)

    # add a factor to the repository as a query result factor
    def add_result_factor(self, factor):
        # adds the factor and return its position
        return self.factors_repository.add(factor, removable=False)

    # gets the number of operations stored
    def get_size(self):
        return len(self.repository)

    # gets the operation for a certain index
    def get_operation(self, index):
        return self.repository[index]

    # gets the number of repeated operations in the repository
    def get_repetitions(self):
        repetitions = 0
        for operation in self.repository.values():
            if operation.repeated == True:
                repetitions += 1

        # return the number of repetitions
        return repetitions

    # gets the maximum size of memory required during the
    # computation
    def get_max_factors_size(self):
        max_size = 0
        for operation in self.repository.values():
            if max_size < operation.getCost()[1]:
                max_size = operation.getCost()[1]

        # return max_size
        return max_size

    # gets the real size of memory required during the
    # computation
    def get_real_factors_size(self):
        real_size = 0
        for operation in self.repository.values():
            if real_size < operation.getCost()[0]:
                real_size = operation.getCost()[0]

        # return real_size
        return real_size

    # return the list of indexes containing the last operation
    # related to each potential
    def get_factors_removable_time(self):
        indexes = []

        # for each factor in the repository of factors, determine
        # the last operation where it is used
        for factorIndex in range(0, self.factors_repository.size()):
            tuple = self.factors_repository.get_factor_data(factorIndex)
            # adds a tuple with factor index and index of last operation
            # where the factor is employed. The last part of the tuple
            # is removable flag
            indexes.append((factorIndex, self.last_operation_for_factor(tuple[1]),
                            tuple[2]))

        # return the list of indexes
        return indexes

    # groups factors according to removable times. The method produce a
    # list of tuples with (operation index - list of factors)
    def group_removable_times(self, factor_op):
        stages = []
        for operationIndex in range(0, self.get_size()):
            # creates an empty list for the factors to be removed
            # after operation in operationIndex
            removableFactors = []

            # looks for all the entries in factor_op where the
            # second element matches operationIndex
            for tupleIndex in range(0, len(factor_op)):
                # check the second part of the tuple (the index of operation)
                # and if this is the same as operationIndex, then add the
                # factor index (first part of tuple) into the actual stage
                # list
                tuple = factor_op[tupleIndex]

                # add the factor if times matches and the factor is removable
                if tuple[1] == operationIndex and tuple[2] == True:
                    removableFactors.append(tuple[0])

            # at the end adds a new tuple to stages
            stages.append((operationIndex, removableFactors))

        # return states
        return stages

    # return the index of the last operation where the factor
    # is involved
    def last_operation_for_factor(self, factor):
        last_index = 0
        indexes = []
        for index in range(0, len(self.repository)):
            operation = self.repository[index]
            op_phi1 = self.factors_repository.get_factor(operation.phi1_index)
            op_result = self.factors_repository.get_factor(operation.result_index)
            if id(op_phi1) == id(factor):
                last_index = index
            else:
                if id(op_result) == id(factor):
                    last_index = index
                else:
                    if operation.phi2_index != -1:
                        op_phi2 = self.factors_repository.get_factor(operation.phi2_index)
                        if id(op_phi2) == id(factor):
                            last_index = index

        # return last index
        return last_index

    # gets the list of removable factors one this operation is performed
    def get_removable_factors(self, operation_index):
        # delegates on factor repository this operation
        return self.factors_repository.get_removable_factors(operation_index)

    # remove the factors passed as argument as a list
    # def remove_factors(self, factors_list):
    def remove_factors(self):
        # propagate factors remove to factors repository
        # self.factors_repository.remove(factors_list)
        self.factors_repository.remove_with_time(self.counter)

    # get the counter of ractor removals
    def get_removed(self):
        return self.factors_repository.get_removed();

    # return a string with the information of the repository
    def __str__(self):
        # shows information about repository of operations
        result = "---------------------- operations repo ----------------------\n"
        result += "sequence of operations ...................\n"
        result += "number of operations: " + str(len(self.repository)) + "\n"
        for key in self.repository:
            result += "operation " + str(key) + ": \n"
            if self.repository[key] is not None:
                entry = self.repository[key]
                result += entry.__str__()
            else:
                result += "None"
        result += "sequence of factors ...................\n"
        result += "number of factors: " + str(len(self.factors_repository.factors))
        result += self.factors_repository.__str__()
        result += "-------------------------------------------------------------\n"

        # return result
        return result
