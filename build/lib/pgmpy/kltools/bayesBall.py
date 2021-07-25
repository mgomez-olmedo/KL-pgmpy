import pgmpy.base.DirectedGraph


class BayesBallNode:

    def __init__(self, variable, parents, children):
        self.inJ = False
        self.inK = False
        self.top = False
        self.down = False
        self.visited = False
        self.fromParent = False
        self.scheduled = False
        self.parents = parents
        self.children = children

    def reset_flags(self):
        self.inJ = False
        self.inK = False
        self.top = False
        self.down = False
        self.visited = False
        self.fromParent = False
        self.scheduled = False


class BayesBall:

    def __init__(self, graph):
        self.nodes = []
        self.dict = {}
        for node in graph.Nodes():
            bbNode = BayesBallNode(node, graph.get_parents(node), graph.get_children(node))
            self.dict[node] = bbNode
            self.nodes.append(bbNode)

    def get_relevant(self, target, observed):
        for node in self.nodes:
            node.resetFlags()

        for targetVariable in target:
            self.dict(targetVariable).inJ = True

        for observedVariable in observed:
            self.dict(observedVariable).inK = True

        # get nodes inJ
        inJ = [node for node in self.nodes if node.inJ == True]

        # call bounce on inJ
        self.bounce(inJ)

        # gets and return nodes marked in top
        return filter(lambda node: node.top == True, self.nodes)

    def bounce(self, inJ):
        if len(inJ) != 0:
            node = self.pop(0)

            toVist = []
            if node.fromParent == True:
                toVisit = self.receiveBallFromParent(node, inJ)
            else:
                toVisit = self.receiveBallFromChild(node, inJ)

            # new call to bounce with toVisit
            self.bounce(toVist)

    def receiveFromChild(self, current, toVisit):
        result = toVisit

        current.visited = True
        if current.inK == False:
            if current.top == False:
                current.top = True
                for parent in current.parents:
                    if parent.scheduled == False:
                        parent.scheduled = True
                        parent.fromParent = False
                        result.append(parent)

            if current.down == False:
                current.down = True
                for child in current.children:
                    if child.scheduled == False:
                        child.scheduled = True
                        child.fromParent = True
                        result.append(child)

        return result

    def receiveFromParent(self, current, toVisit):
        result = toVisit
        current.visited = True
        if current.inK == True:
            if current.top == False:
                current.top = True
                for parent in current.parents:
                    if parent.scheduled == False:
                        parent.fromNode = False
                        parent.scheduled = True
                        result.append(parent)
        else:
            if current.down == False:
                current.down == True
                for child in current.children:
                    if child.scheduled == False:
                        child.fromParent == True
                        child.scheduled == True
                        result.append(child)

        return result
