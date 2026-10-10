from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NewsClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False

    def train(self, tfidf_matrix, categories):
        if len(categories) < 10: return 0.0
        X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, categories, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        acc = accuracy_score(y_test, self.model.predict(X_test))
        self.model.fit(tfidf_matrix, categories)
        self.is_trained = True
        return acc

    def predict(self, tfidf_vectorizer, processed_text):
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        vec = tfidf_vectorizer.transform([processed_text])
        return self.model.predict(vec)[0]
