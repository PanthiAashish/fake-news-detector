from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def get_vectorizer(X_train, X_test):
    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit the vectorizer on training data and transform both train and test sets
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Save the vectorizer for future use (optional)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    return X_train_tfidf, X_test_tfidf, vectorizer
