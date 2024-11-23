from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def train_models(X_train, y_train):
    # Initialize models
    naive_bayes = GaussianNB()
    knn = KNeighborsClassifier()

    # Hyperparameter tuning for KNN
    param_grid = {'n_neighbors': range(3, 21, 2)}
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['n_neighbors']

    # Train models
    naive_bayes.fit(X_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    return naive_bayes, knn
