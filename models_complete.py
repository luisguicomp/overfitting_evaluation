from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
import numpy as np
# Base de dados sintética
X, y = make_classification(n_samples=5000, n_features=30, n_informative=5, n_redundant=10, random_state=1)
# dividindo a base de dados em treno e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# listas das avaliações
train_scores, test_scores, overfitting_score = list(), list(), list()

names = ['KNN', 'SVM', 'RF', 'MLP', 'NB']
models = [KNeighborsClassifier(n_neighbors=5), SVC(), RandomForestClassifier(), MLPClassifier(max_iter=500), GaussianNB()]

i = 0
for m in models:
	model = m
	model.fit(X_train, y_train)

	train_yhat = model.predict(X_train)
	train_acc = accuracy_score(y_train, train_yhat)
	train_scores.append(round(train_acc,2))

	test_yhat = model.predict(X_test)
	test_acc = accuracy_score(y_test, test_yhat)
	test_scores.append(round(test_acc,2))

	#Overfitting baseado na razão entre teste e treino
	overfit_acc = test_acc/train_acc
	overfitting_score.append(round(overfit_acc,2))

	print(names[i])
	print('treino: %.3f, teste: %.3f , overfitting: %.3f' % (train_acc, test_acc, overfit_acc))
	i = i+1

x = np.arange(len(names))
width = 0.30

fig, ax = pyplot.subplots()
rects1 = ax.bar(x - width/2, train_scores, width, label='Treino')
rects2 = ax.bar(x + width/2, test_scores, width, label='Teste')
rects3 = ax.bar(x + 1.5*width, overfitting_score, width, label='Overfitting')

ax.set_ylabel('Overfitting - Acurácia')
ax.set_title('Avaliação de overfitting')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend(bbox_to_anchor=(1, 1))

ax.bar_label(rects1, padding=2)
ax.bar_label(rects2, padding=2)
ax.bar_label(rects3, padding=2)

fig.tight_layout()

pyplot.show()
