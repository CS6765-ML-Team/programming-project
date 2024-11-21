class BinaryDecisionTree:
    """ BinaryDecisionTree represents a binary tree
    implementation used for the ID3 decision tree
    training algorithm. Each tree contains 4 parameters:
        - Feature Index:    The current feature index the tree splits on
        - Value:            The value of Feature the determins the split
        - True Branch:      The branch when Feature > Value
        - False Branch:     The branch when Feature <= Value
    """
    def __init__(self, feature_idx, value):
        self.feature_idx = feature_idx
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

        feature_value = data_row[self.feature_idx]
        if (feature_value > self.value):
            return self.true_branch.traverse(data_row)
        else:
            return self.false_branch.traverse(data_row)

    def print_tree(self, features, level=0):
        if (self.false_branch is not None):
            self.false_branch.print_tree(level + 1)
        print(' ' * 4 * level + '->' + f"{features.columns[self.feature_idx]}: {self.value}")
        if (self.true_branch is not None):
            self.true_branch.print_tree(level + 1)