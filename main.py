from src.load_data import load_email_data
from src.preprocess import preprocess_text
from src.train_decision_tree import train_decision_tree
from src.evaluate import evaluate_model
from src.plot_tree import plot_decision_tree

def main():
    # Load dataset
    df = load_email_data()

    # Preprocess and split
    X_train, X_test, y_train, y_test, vectorizer = preprocess_text(df)

    # Train model
    model = train_decision_tree(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Plot tree
    plot_decision_tree(model, vectorizer)

if __name__ == "__main__":
    main()
