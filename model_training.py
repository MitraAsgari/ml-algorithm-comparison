from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import time

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, f1, train_time

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
}

results = {}
for name, model in models.items():
    accuracy, f1, train_time = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    results[name] = {'accuracy': accuracy, 'f1_score': f1, 'train_time': train_time}

print(results)
