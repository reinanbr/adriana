 <div align='center'>

 <img  height='180px' width='420px' src='https://github.com/reinanbr/adriana/blob/main/logo.jpg?raw=true'>


<p> A powerfull lib, for development in machine learning</p>
<a href='#'><img alt="CodeFactor Grade" src="https://img.shields.io/codefactor/grade/github/reinanbr/dreams?logo=codefactor">
</a><img alt="CircleCI" src="https://img.shields.io/circleci/build/github/reinanbr/dreams">
<img alt="Code Climate maintainability" src="https://img.shields.io/codeclimate/maintainability-percentage/reinanbr/dreams">
<!-- 
<br/>
<a href='https://pypi.org/project/dreams/'><img src='https://img.shields.io/pypi/v/dreams'></a>
<a href='#'><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/dreams"></a>
<br/>
<img alt="PyPI - License" src="https://img.shields.io/pypi/l/dreams?color=orange">
<img alt="GitHub Pipenv locked Python version" src="https://img.shields.io/github/pipenv/locked/python-version/reinanbr/dreams"> -->


<!-- redes sociais -->
<br/>
<a href='https://instagram.com/reysofts/'><img src='https://shields.io/badge/insta-reysofts-darkviolet?logo=instagram&style=flat'></a>
</div>

<br>

<a href="https://www.buymeacoffee.com/ReinanBr" target="_blank"><img height='30px' widht='100px' src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 30px !important;width: 100px !important;" ></a>

<hr>

## Install:

```sh
$ pip install adrianna -U
```

## Examples:

### Bases

#### Neuro V1 (binary)

```py

from adrianna.neuro.base_v1 import NeuralNetwork
import numpy as np


# Exemplo de uso
if __name__ == "__main__":
    # Dados de entrada e saída
    X = np.array([[3, 1, 2], [1, 24, 5], [2, 42, 5], [2, 23, 3]])
    y = np.array([[1], [0], [1], [0]])
    
    # Criação e treinamento da rede neural
    input_size = X.shape[1]  # Número de colunas das listas
    hidden_size = 4  # Número de neurônios na camada oculta
    output_size = y.shape[1]  # Número de saídas
    
    neural_net = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1)
    neural_net.train(X, y, epochs=10000)
    
    # Fazendo previsões
    predictions = neural_net.predict(X)
    predictions = np.round(predictions).astype(int)
    print("Predictions:")
    print(predictions)

```
Results:

```sh
...
Epoch 8000, Loss: 0.2617957
Epoch 8100, Loss: 0.1830651
Epoch 8200, Loss: 0.1800935
Epoch 8300, Loss: 0.2585192
Epoch 8400, Loss: 0.3310709
Epoch 8500, Loss: 0.3215849
Epoch 8600, Loss: 0.1803035
Epoch 8700, Loss: 0.1802555
Epoch 8800, Loss: 0.1807692
Epoch 8900, Loss: 0.3312975
Epoch 9000, Loss: 0.1800601
Epoch 9100, Loss: 0.2642676
Epoch 9200, Loss: 0.1930688
Epoch 9300, Loss: 0.3279387
Epoch 9400, Loss: 0.1871483
Epoch 9500, Loss: 0.1809427
Epoch 9600, Loss: 0.2635577
Epoch 9700, Loss: 0.1855325
Epoch 9800, Loss: 0.1796770
Epoch 9900, Loss: 0.1800897

Predictions:
[[1]
 [0]
 [1]
 [1]]


```

<img src='https://reysofts.com.br/engine/libs/save_table_access_libs.php?lib_name=adrianna'>