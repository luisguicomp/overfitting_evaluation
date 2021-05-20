from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# Base de dados sintética
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=15, random_state=1)
# dividindo a base de dados em treno e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)