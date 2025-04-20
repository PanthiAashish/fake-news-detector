from src.preprocess import load_and_clean_data
from src.vectorizer import get_vectorizer
from src.models import train_models
from src.evaluate import evaluate_model
from src.visualize import generate_wordclouds

def main():
    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_clean_data()

    # 2. Vectorize using TF-IDF
    X_train_tfidf, X_test_tfidf, vectorizer = get_vectorizer(X_train, X_test)

    # 3. Train models
    nb_model, lr_model = train_models(X_train_tfidf, y_train)

    # 4. Evaluate models
    evaluate_model("Naive Bayes", nb_model, X_test_tfidf, y_test)
    evaluate_model("Logistic Regression", lr_model, X_test_tfidf, y_test)

    # 5. Generate WordClouds
    generate_wordclouds()

if __name__ == "__main__":
    main()
