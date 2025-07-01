from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train, max_depth=10):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model
