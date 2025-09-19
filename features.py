# features.py
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def apply_pca(X_train, X_test, n_components=5):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def select_features(X_train, y_train, X_test, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_new = selector.fit_transform(X_train, y_train)
    X_test_new = selector.transform(X_test)
    return X_train_new, X_test_new, selector
