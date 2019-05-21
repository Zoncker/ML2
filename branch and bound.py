import random
from graphviz import Digraph
import queue
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets

iris = datasets.load_iris()


def criterion_function(features):

    X_train, X_test, y_train, y_test = train_test_split(iris.data[:, features],
                                                        iris.target, test_size=0.4, random_state=0)
    # clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")  # construct a decision tree.
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    # result = knn.score(X_test, y_test)
    result = cross_val_score(knn, iris.data[:, features], iris.target, cv=5)

    return min(result)


# def isMonotonic(features):
#     features = sorted(features)
#
#     # Generate the powerset of the features
#     powerset = []
#     for i in range(1, len(features) + 1):
#         subset = itertools.combinations(features, i)
#         powerset.extend(subset)
#
#     # For all possible subset pairs, check if monotonicity is satisfied
#     # print (powerset)
#     for i, item1 in enumerate(powerset):
#         for item2 in powerset[i + 1:]:
#             if (set(item1).issubset(set(item2)) and (
#                     criterion_function(list(item1)) > criterion_function(list(item2)))):
#                 return False
#
#     return True


class tree_node(object):
    def __init__(self, value, features, preserved_features, level):
        self.branch_value = value
        self.features = features
        self.preserved_features = preserved_features
        self.level = level
        self.name = []
        self.index = None
        self.children = []
        self.J = None


flag = True
J_max = -1
result_node = None


def branch_and_bound(root, D, d):
    global flag
    global J_max
    global result_node

    # Compute the criterion function
    root.J = criterion_function(root.features)

    # Stop building children for this node, if J <= J_max
    if flag == False and root.J <= J_max:
        return

    # If this is the leaf node, update J_max, result_node and return
    if root.level == D - d:
        if flag == True:
            J_max = root.J
            flag = False
            result_node = root

        elif root.J > J_max:
            J_max = root.J
            result_node = root

        return

    # Compute the number of branches for this node
    no_of_branches = (d + 1) - len(root.preserved_features)

    # Generate the branches
    branch_feature_values = sorted(
        random.sample(list(set(root.features) - set(root.preserved_features)), no_of_branches))

    # Iterate on the branches, and for each branch, call branch_and_bound recursively
    for i, branch_value in enumerate(branch_feature_values):
        child = tree_node(branch_value, [value for value in root.features if value != branch_value], \
                          root.preserved_features + branch_feature_values[i + 1:], root.level + 1)

        root.children.append(child)

        branch_and_bound(child, D, d)


def give_indexes(root):
    bfs = queue.Queue(maxsize=40)

    bfs.put(root)
    index = -1
    while bfs.empty() == False:
        node = bfs.get()
        node.index = index
        index += 1
        for child in node.children:
            bfs.put(child)


def display_tree(node, dot_object, parent_index):
    iris = datasets.load_iris()
    # asd = slice(node.features)
    for i in node.features:
        node.name.append(iris.feature_names[i])

    # Create node in dob_object, for this node
    dot_object.node(str(node.index),
                    "Features = " + str(node.name) + "\nJ(Features) = " + str(node.J) + "\nPreserved = " + str(
                        node.preserved_features))

    # If this is not the root node, create an edge to its parent
    if node.index != -1:
        dot_object.edge(str(parent_index), str(node.index), label=str(node.branch_value))

    # Base case, when the node has no children, return
    if len(node.children) == 0:
        return

    # Recursively call display_tree for all the childern of this node
    for child in reversed(node.children):
        display_tree(child, dot_object, node.index)


# def parse_features(features_string):
#     return sorted([float(str) for str in features_string.split(',')])

def main():
    features = [0, 1, 2, 3]
    D = len(features)
    d = 2
    flag = 0

    # Create the root tree node
    root = tree_node(-1, features, [], 0)

    # Call branch and bound on the root node, and recursively construct the tree
    branch_and_bound(root, D, d)

    # Give indexes(numbers) for nodes of constructed tree in BFS order (used for Graphviz)
    give_indexes(root)

    # Display the constructed tree using python graphviz
    print("Plotting branch and bound tree...")
    dot = Digraph(comment="Branch and Bound Feature selection")
    dot.format = "png"
    display_tree(root, dot, -1)
    dot.render("bb_tree", view=True)
    print("Plotting finished...")

    # Print the result
    print("------")
    print("Output")
    print("------")
    print("Features considered = {}".format(result_node.features))
    print("Criteriion function value = {}".format(result_node.J))


if __name__ == "__main__":
    main()