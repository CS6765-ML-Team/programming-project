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

    def traverse(self, data_row):
        if (self.true_branch is None and self.false_branch is None):
            return self.value

        feature_value = data_row[self.feature]
        if (feature_value > self.value):
            return self.true_branch.traverse(data_row)
        else:
            return self.false_branch.traverse(data_row)

    def __str__(self):
        return f"{self.feature}: {self.value}\n\t\tFalse Branch: {self.false_branch}\n---------\n\t\tTrue Branch: {self.true_branch}\n"

    def print_tree(self, level=0):
        if (self.false_branch is not None):
            self.false_branch.print_tree(level + 1)
        print(' ' * 4 * level + '->' + f"{self.feature}: {self.value}")
        if (self.true_branch is not None):
            self.true_branch.print_tree(level + 1)