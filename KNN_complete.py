# evaluate knn performance on train and test sets with different numbers of neighbors
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
# Base de dados sintética
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=15, random_state=1)
# dividindo a base de dados em treno e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# listas das avaliações
train_scores, test_scores, overfitting_score = list(), list(), list()

values = [i for i in range(1, 31)]

for i in values:
	# Algoritmo KNN
	model = KNeighborsClassifier(n_neighbors=i)

	model.fit(X_train, y_train)

	train_yhat = model.predict(X_train)
	train_acc = accuracy_score(y_train, train_yhat)
	train_scores.append(train_acc)

	test_yhat = model.predict(X_test)
	test_acc = accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
	#Overfitting baseado na razão entre teste e treino
	overfit_acc = test_acc/train_acc
	overfitting_score.append(overfit_acc)

	print('>%d, treino: %.3f, teste: %.3f , overfitting: %.3f' % (i, train_acc, test_acc, overfit_acc))

pyplot.plot(values, train_scores, '-o', label='Treino')
pyplot.plot(values, test_scores, '-o', label='Teste')
pyplot.plot(values, overfitting_score, '-o', label='Overfitting')

pyplot.legend()
pyplot.show()