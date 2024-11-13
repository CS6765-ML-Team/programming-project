class BinaryDecisionTree:
    """ BinaryDecisionTree represents a binary tree
    implementation used for the ID3 decision tree
    training algorithm. Each tree contains 4 parameters:
        - Feature:      The current feature the tree splits on
        - Value:        The value of Feature the determins the split
        - True Branch:  The branch when Feature >= Value
        - False Branch: The branch when Feature < Value
    """
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value
        self.true_branch = None
        self.false_branch = None

    def add_true(self, branch):
        self.true_branch = branch

    def add_false(self, branch):
        self.false_branch = branch

class DecisionLeaf:
    def __init__(self, target):
        self.target = target

    def print_decision(self):
        print("Classification: {}\n".format(self.target))