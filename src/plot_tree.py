import matplotlib.pyplot as plt
from sklearn import tree

def plot_decision_tree(model, vectorizer, max_depth=3):
    feature_names = vectorizer.get_feature_names_out()

    plt.figure(figsize=(20, 10))
    tree.plot_tree(model,
                   max_depth=max_depth,
                   feature_names=feature_names,
                   class_names=["Not Spam", "Spam"],
                   filled=True,
                   fontsize=10)
    plt.title(f"Decision Tree (Top {max_depth} Levels)")
    plt.show()
