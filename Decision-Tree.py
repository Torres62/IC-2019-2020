import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

X = np.array([
    00.001, 00.001, 00.001, 00.001, 00.001,
    00.010, 00.010, 00.010, 00.010, 00.010,
    00.100, 00.100, 00.100, 00.100, 00.100, 00.100,
    01.000, 01.000, 01.000, 01.000, 01.000, 01.000,
    10.000, 10.000, 10.000, 10.000
]).reshape(-1, 1)

y = np.array([
    [13.00], [13.00], [13.30], [14.90], [17.40],
    [08.00], [10.77], [09.57], [15.20], [14.70],
    [02.53], [03.27], [05.97], [06.37], [09.87], [10.63],
    [00.97], [00.40], [03.87], [02.83], [01.83], [00.30],
    [01.53], [05.37], [01.37], [-00.30]
])

#Instaciando regressor de Arvore
reg = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                            min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                            presort=False, random_state=0, splitter='best')

#Split entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Treinando
reg = reg.fit(X_train, y_train)

#Predicao do numero selecionado
numPredict = np.array([0.9]).reshape(-1, 1)
y_pred = reg.predict(numPredict)

#Taxa de acerto com os teste
score = reg.score(X_test, y_test)

#URL PARA VISUALIZAR ARVORE(colocar o código que foi gerado no arquivo .dot), http://www.webgraphviz.com
export_graphviz(reg, out_file='arvore.dot', feature_names=['Dose'])

#PLOT
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X, y, s=1, c='b', marker="s", label='Medição')
ax1.scatter(X_test, y_test, s=10, c='r', marker="o", label='Predição')
plt.xlabel('Dose')
plt.ylabel('Resultado')
plt.title('Machine Learning Aplicado em Estudos da Dor')
plt.savefig('teste-Arvore.png')

#Score e Predição
print(f'Score: {score}')
print(f'Predição com {numPredict} ml : {y_pred}')