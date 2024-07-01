import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        output = self.sigmoid(self.z2)
        return output

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        a1_error = output_delta.dot(self.W2.T)
        a1_delta = a1_error * self.sigmoid_derivative(self.a1)

        self.W2 += self.a1.T.dot(output_delta)
        self.b2 += np.sum(output_delta, axis=0)
        self.W1 += X.T.dot(a1_delta)
        self.b1 += np.sum(a1_delta, axis=0)

    def train(self, X, y, iterations=1000):
        for _ in range(iterations):
            output = self.forward(X)
            self.backward(X, y, output)

# Custom Decision Tree Regressor (DT)
class SimpleDecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth or len(y) < 2:
            return np.mean(y)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.mean(y)

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_mse = float('inf')

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                left_mse = np.var(y[left_indices]) * len(y[left_indices])
                right_mse = np.var(y[right_indices]) * len(y[right_indices])
                mse = (left_mse + right_mse) / len(y)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] <= threshold:
            return self._predict_single(x, left_subtree)
        else:
            return self._predict_single(x, right_subtree)

# Custom Random Forest Regressor (RF)
class SimpleDecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth or len(y) < 2:
            return np.mean(y)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.mean(y)

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_mse = float('inf')

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                left_mse = np.var(y[left_indices]) * len(y[left_indices])
                right_mse = np.var(y[right_indices]) * len(y[right_indices])
                mse = (left_mse + right_mse) / len(y)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] <= threshold:
            return self._predict_single(x, left_subtree)
        else:
            return self._predict_single(x, right_subtree)

class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = [SimpleDecisionTree(max_depth=max_depth) for _ in range(n_trees)]

    def fit(self, X, y):
        for tree in self.trees:
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)
    
# explicit function       
def method(): 
    print("SimpleNeuralNetwork")
    print("SimpleDecisionTree")
    print("SimpleRandomForest")