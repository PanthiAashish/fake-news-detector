from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib
import os
from tensorflow import keras

# Ensure 'models' directory exists
os.makedirs("models", exist_ok=True)

def train_models(X_train, y_train):
    # Train Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Train Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Save the models to disk
    joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
    joblib.dump(lr_model, 'models/logistic_model.pkl')

    return nb_model, lr_model

def train_lstm_model(X_train, y_train):
    # Train LSTM model
    model_lstm = keras.Sequential([
        keras.layers.Embedding(10000, 128),
        keras.layers.LSTM(128),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model_lstm.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    
    model_lstm.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Save the model to disk
    model_lstm.save('models/model_lstm.keras')
    
    return model_lstm
