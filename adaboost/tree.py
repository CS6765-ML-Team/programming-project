# TODO: Maybe only implement this as a binary tree with less than and greater
# than splits of the decision tree values?
class DecisionTree:
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

class DecisionLeaf:
    def __init__(self, target):
        self.target = target

    def print_decision(self):
        print("Classification: {}\n".format(self.target))