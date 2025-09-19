# models.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

def get_supervised_models():
    return {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(),
        'svm': SVC(probability=True)
    }

def get_unsupervised_models():
    return {
        'kmeans': KMeans(n_clusters=2, random_state=42)
    }
