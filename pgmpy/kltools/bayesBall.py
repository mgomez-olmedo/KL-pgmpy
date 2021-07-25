import pgmpy.base.DAG

# class BayesBallNode for determining relevance of nodes
# for posterior computations
class BayesBallNode:

    # class constructor
    def __init__(self, variable, parents, children):
        self.variable = variable
        self.inJ = False
        self.inK = False
        self.top = False
        self.down = False
        self.visited = False
        self.from_parent = False
        self.from_child = False
        self.scheduled = False
        self.parents = parents
        self.children = children

    # reset flags for starting a new computation
    def reset_flags(self):
        self.inJ = False
        self.inK = False
        self.top = False
        self.down = False
        self.visited = False
        self.from_parent = False
        self.from_child = False
        self.scheduled = False

    # gets a string with information about nodes for
    # bayes ball computation
    def __str__(self):
        str = "BayesBallNode for " + self.variable + "\n"
        str = str + "inJ: " + self.inJ.__str__() + "\n"
        str = str + "inK: " + self.inK.__str__() + "\n"
        str = str + "top: " + self.top.__str__() + "\n"
        str = str + "down: " + self.down.__str__() + "\n"
        str = str + "visited: " + self.visited.__str__() + "\n"
        str = str + "from_parent: " + self.from_parent.__str__() + "\n"
        str = str + "from_child: " + self.from_child.__str__() + "\n"
        str = str + "scheduled: " + self.scheduled.__str__() + "\n"
        str = str + "---------------------------------------" + "\n"
        return str

# class BayesBall for making computations of relevance and
# irrelevance of variables
class BayesBall:

    # class constructor
    def __init__(self, model):
        # list of nodes of class BayesBallNode
        self.nodes = []

        # dictionary for getting access to nodes with variable
        # names as keys
        self.dict = {}

        # loop on model variables
        for node in model.nodes():
            # makes a new BayesBallNode
            bbNode = BayesBallNode(node, model.get_parents(node),
                                   model.get_children(node))

            # add it ti dictionary and list of nodes
            self.dict[node] = bbNode
            self.nodes.append(bbNode)

    # computes the relevant nodes for a computation on
    # a set of target variables
    def get_relevant(self, target, observed = []):
        # reset flags for starting a fresh computation
        for node in self.nodes:
            node.reset_flags()

        # mark target variables
        for target_variable in target:
            self.dict[target_variable].inJ = True
            self.dict[target_variable].scheduled = True
            self.dict[target_variable].from_child = True

        # mark observed variables
        for observed_variable in observed:
            self.dict[observed_variable].inK = True

        # get nodes (BayesBallNodes) inJ
        inJ = [value for (key,value) in self.dict.items() if value.inJ == True]

        # call bounce on inJ nodes
        self.__bounce(inJ)

        # gets and return nodes marked in top
        bbnodes = filter(lambda node: node.top == True, self.nodes)
        return list(map(lambda bbnode: bbnode.variable, bbnodes))

    # main method for bouncing balls between nodes
    # private method
    def __bounce(self, inJ):
        # gets first node to consider
        if len(inJ) != 0:
            node = inJ.pop(0)

            # initialize toVist
            to_visit = []
            if node.from_parent == True:
                to_visit = self.__receive_from_parent(node, inJ)

            if node.from_child == True:
                to_visit = self.__receive_from_child(node, inJ)

            # new call to bounce with toVisit
            self.__bounce(to_visit)

    # method for passing balls comming from children
    # private method
    def __receive_from_child(self, current, to_visit):
        # initialize result to
        result = to_visit

        # mark node as visited
        current.visited = True

        # treatment for non observed nodes
        if current.inK == False:
            # propagate to parents
            if current.top == False:
                current.top = True
                for parent in current.parents:
                    # gets BBNode for parent
                    parentBBNode = self.dict[parent]
                    #if not parentBBNode.scheduled:
                    parentBBNode.scheduled = True
                    parentBBNode.from_child = True
                    result.append(parentBBNode)

            # propagate to children
            if current.down == False:
                current.down = True
                for child in current.children:
                    # get BBNode for child
                    childBBNode = self.dict[child]
                    #if not childBBNode.scheduled:
                    childBBNode.scheduled = True
                    childBBNode.from_parent = True
                    result.append(childBBNode)

        # return the list of nodes to be visited
        return result

    # mothod for passing balls comming from parents
    # private method
    def __receive_from_parent(self, current, to_visit):
        # initialize result to the list of nodes to visit
        # passed as argument
        result = to_visit

        # mark node as visited
        current.visited = True

        # treatment of observed nodes
        if current.inK == True:
            # propagate to children
            if current.top == False:
                current.top = True
                for parent in current.parents:
                    # get BBNode for parent
                    parentBBNode = self.dict[parent]
                    #if not parentBBNode.scheduled:
                    parentBBNode.from_child = True
                    parentBBNode.scheduled = True
                    result.append(parentBBNode)
        # treatment of non observed nodes
        else:
            # propagate on children
            if current.down == False:
                current.down = True
                for child in current.children:
                    # get child BBNode
                    childBBNode = self.dict[child]
                    #if not childBBNode.scheduled:
                    childBBNode.from_parent = True
                    childBBNode.scheduled = True
                    result.append(childBBNode)

        return result

    # gets a str with the information of BayesBall nodes
    def __str__(self):
        str = "Bayes Ball object content: \n"
        for node in self.nodes:
            str += node.__str__() + "\n"
        return str
