# ML2

# Скользящий контроль

Скользящий контроль или кросс-проверка или кросс-валидация (cross-validation, CV) — процедура эмпирического оценивания обобщающей способности алгоритмов, обучаемых по прецедентам.

Фиксируется некоторое множество разбиений исходной выборки на две подвыборки: обучающую и контрольную. Для каждого разбиения выполняется настройка алгоритма по обучающей подвыборке, затем оценивается его средняя ошибка на объектах контрольной подвыборки. Оценкой скользящего контроля называется средняя по всем разбиениям величина ошибки на контрольных подвыборках.

Рассматривается задача обучения с учителем.

Пусть  X — множество описаний объектов,  Y — множество допустимых ответов.

Задана конечная выборка прецедентов ![](https://latex.codecogs.com/svg.latex?X%5EL%20%3D%20%28x_i%2Cy_i%29_%7Bi%3D1%7D%5EL%20%5Csubset%20X%5Ctimes%20Y.)

Задан алгоритм обучения — отображение  ![](https://latex.codecogs.com/svg.latex?%5Cmu), которое произвольной конечной выборке прецедентов ![](https://latex.codecogs.com/svg.latex?X%5Em) ставит в соответствие функцию (алгоритм) ![](https://latex.codecogs.com/svg.latex?a%3A%5C%3AX%5Cto%20Y).

Качество алгоритма a оценивается по произвольной выборке прецедентов ![](https://latex.codecogs.com/svg.latex?X%5Em) с помощью функционала качества ![](https://latex.codecogs.com/svg.latex?Q%28a%2CX%5Em%29). Для процедуры скользящего контроля не важно, как именно вычисляется этот функционал. Как правило, он аддитивен по объектам выборки:

![](https://latex.codecogs.com/svg.latex?Q%28a%2CX%5Em%29%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bx_i%5Cin%20X%5Em%7D%20%5Cmathcal%7BL%7D%28a%28x_i%29%2Cy_i%29%2C)

### Полный скользящий контроль (complete CV)

Оценка скользящего контроля строится по всем ![](https://latex.codecogs.com/svg.latex?N%3DC_L%5Ek)  разбиениям. В зависимости от **k** (длины обучающей выборки) различают:

Частный случай при **k=1** — контроль по отдельным объектам (leave-one-out CV);

### Случайные разбиения

Разбиения ![](https://latex.codecogs.com/svg.latex?n%3D1%2C%5Cldots%2CN) выбираются случайно, независимо и равновероятно из множества всех ![](https://latex.codecogs.com/svg.latex?C_L%5Ek) разбиений. 

### q-fold CV

Выборка случайным образом разбивается на q непересекающихся блоков одинаковой (или почти одинаковой) длины ![](https://latex.codecogs.com/svg.latex?k_1%2C%5Cldots%2Ck_q%3A), ![](https://latex.codecogs.com/svg.latex?k_1&plus;%5Cdots&plus;k_q%20%3D%20L.)  Каждый блок по очереди становится контрольной подвыборкой, при этом обучение производится по остальным q-1 блокам. Критерий определяется как средняя ошибка на контрольной подвыборке:

![](https://latex.codecogs.com/svg.latex?CV%28%5Cmu%2CX%5EL%29%3D%5Cfrac1q%20%5Csum_%7Bn%3D1%7D%5Eq%20Q%20%5Cbigl%28%20%5Cmu%28X%5EL%5Csetminus%20X%5E%7Bk_n%7D_n%29%2C%20X%5E%7Bk_n%7D_n%20%5Cbigr%29.)

### r×q-fold CV
Контроль по q блокам (q-fold CV) повторяется r раз. Каждый раз выборка случайным образом разбивается на q непересекающихся блоков. Этот способ наследует все преимущества q-fold CV, при этом появляется дополнительная возможность увеличивать число разбиений.


| CV Type\ML Algorithm | KNN\wine | SVC\wine |
|----------------------|----------|----------|
| Complete CV          | 0.9154   | 0.8642   |
| Random permut        | 0.8635   | 0.8205   |
| r x q-fold           | 0.8842   | 0.7986   |
| Bootstrap            | 0.8743   | 0.7895   |
| LOO                  | 0.8921   | 0.8423   |


# AdaBoost

Алгоритм AdaBoost - мета-алгоритм, в процессе обучения строит композицию из базовых алгоритмов обучения для улучшения их эффективности. AdaBoost является алгоритмом адаптивного бустинга в том смысле, что каждый следующий классификатор строится по объектам, которые плохо классифицируются предыдущими классификаторами.

AdaBoost вызывает слабый классификатор в цикле. После каждого вызова обновляется распределение весов, которые отвечают важности каждого из объектов обучающего множества для классификации. На каждой итерации веса каждого неверно классифицированного объекта возрастают, таким образом новый классификатор «фокусирует своё внимание» на этих объектах.


Рассмотрим задачу классификации на два класса, ![](https://latex.codecogs.com/svg.latex?Y%3D%5C%7B-1%2C&plus;1%5C%7D.) Допустим, что базовые алгоритмы ![](https://latex.codecogs.com/svg.latex?b_1%2C%20%5Cdots%2C%20b_T) также возвращают только два ответа -1 и +1. ![](https://latex.codecogs.com/svg.latex?W%5El%20%3D%20%28w_1%2C%5Cdots%20w_l%29) — вектор весов объектов.
![](https://latex.codecogs.com/svg.latex?Q%28b%2CW%5El%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bl%7Dw_i%5By_i%20b%28x_i%29%20%3C%200%5D) — стандартный функционал качества алгоритма классификации b.

Задачу оптимизации параметра ![](https://latex.codecogs.com/svg.latex?%5Calpha_t) решаем аналитически, аппроксимируя пороговую функцию потерь ![](https://latex.codecogs.com/svg.latex?%5Bz%20%3C%200%5D) с помощью экспоненты ![](https://latex.codecogs.com/svg.latex?E%28z%29%20%3D%20%5Cexp%28-z%29.)

Дано: ![](https://latex.codecogs.com/svg.latex?X%5El) - обучающая выборка;

![](https://latex.codecogs.com/svg.latex?b_1%2C%20%5Cdots%2C%20b_T) - базовые алгоритмы классификации;

1. Инициализация весов объектов: ![](https://latex.codecogs.com/svg.latex?%5Cw_i%20%3D%201/l%2C%20i%20%3D%201%2C%5Cdots%2C%20l%3B)
2. Для всех ![](https://latex.codecogs.com/svg.latex?t%3D1%2C%5Cdots%2C%20T%2C) пока не выполнен критерий останова.

2.1 Находим классификатор ![](https://latex.codecogs.com/svg.latex?b_%7Bt%7D%3A%20X%20%5Cto%20%5C%7B-1%2C&plus;1%5C%7D) который минимизирует взвешенную ошибку классификации;
![](https://latex.codecogs.com/svg.latex?b_t%20%3D%20%5Carg%20%5Cmin_b%20Q%28b%2CW%5El%29%3B)

2.2 Пересчитываем кооэффициент взвешенного голосования для алгоритма классификации ![](https://latex.codecogs.com/svg.latex?b_t):
    
![](https://latex.codecogs.com/svg.latex?%5Calpha_t%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Cln%5Cfrac%7B1%20-%20Q%28b%2CW%5El%29%7D%7BQ%28b%2CW%5El%29%7D%3B)

2.3 Пересчет весов объектов: ![](https://latex.codecogs.com/svg.latex?w_i%20%3D%20w_i%20%5Cexp%7B%28-%5Calpha_t%20y_i%20b_t%28x_i%29%29%7D%2C%20i%20%3D%201%2C%5Cdots%2C%20l);

2.4 Нормировка весов объектов: ![](https://latex.codecogs.com/svg.latex?w_0%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7Bl%7Dw_j%3B%20w_i%20%3D%20w_i/w_0%2C%20i%20%3D%201%2C%5Cdots%2C%20l);

4. Возвращаем ![](https://latex.codecogs.com/svg.latex?a%28x%29%20%3D%20sign%20%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7BT%7D%20%5Calpha_i%20b_i%28x%29%5Cright%29):

### Реализация

```python
def AdaBoost(X, y, M=10, learning_rate=1):
    # инициализация утиль списков
    N = len(y)
    estimator_list, y_predict_list, estimator_error_list, estimator_weight_list, sample_weight_list = [], [], [], [], []
    # инициализация весов
    sample_weight = np.ones(N) / N
    sample_weight_list.append(sample_weight.copy())
    for m in range(M):
        # Обучаем слабый классификатор
        estimator = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)

        # количество мисклассификаций
        incorrect = (y_predict != y)
        # ошибка эстимейтора
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        # Усиливаем веса эстимейтора
        estimator_weight = learning_rate * np.log((1. - estimator_error) / estimator_error)
        # усиливаем веса объектов
        sample_weight *= np.exp(estimator_weight * incorrect * ((sample_weight > 0) | (estimator_weight < 0)))

        # Сохраняем данные итерации
        estimator_list.append(estimator)
        y_predict_list.append(y_predict.copy())
        estimator_error_list.append(estimator_error.copy())
        estimator_weight_list.append(estimator_weight.copy())
        sample_weight_list.append(sample_weight.copy())

    # конвертим в ndarray для удобства
    estimator_list = np.asarray(estimator_list)
    y_predict_list = np.asarray(y_predict_list)
    estimator_error_list = np.asarray(estimator_error_list)
    estimator_weight_list = np.asarray(estimator_weight_list)
    sample_weight_list = np.asarray(sample_weight_list)
    
    #предсказания
    preds = (np.array([np.sign((y_predict_list[:, point] * estimator_weight_list).sum()) for point in range(N)]))
    print('Accuracy = ', (preds == y).sum() / N)

    return estimator_list, estimator_weight_list, sample_weight_list
```
Используем игрушечные данные, количество итераций **М=10**, слабый классификатор - DecisionTreeClassifier с глубиной 1. Граница между красными и синими точками нелинейна, но алгоритм работает очень хорошо:

![](ada2.png) 

Покажем цепочку слабых обучений и примеры весов. Первые 9 слабых обучений представлены ниже, точки масштабированы согласно их весов на каждой итерации:

![](ada33.png)

### Влияние темпа обучения L и количества слабых классификаторов M:

Уменьшение L делает коэфф-ты ![](https://latex.codecogs.com/svg.latex?%5Calpha_m) меньше, что уменьшает амплитуду весов объектов на каждом шаге, т.к ![](https://latex.codecogs.com/svg.latex?w_i%20%5Cleftarrow%20w_i%20e%5E%7B%20%5Calpha_m%20%5Cmathcal%7BI%7D...%7D). Это приводит к меньшим изменениям взвешенных эл-тов, следовательно, к меньшей разнице между разделениями слабых классификаторов.

Увеличение M увеличивает кол-во итераций, и позволяет весам объектов нарастить большую амплитуду, что приводит к увеличению кол-ва слабых классификаторов в конце, больше вариаций в разделяющих поверхностях.

Следовательно, логично предположить компромисс между двумя параметрами (увеличением одного и уменьшением другого)

![](ada_f.png)

Слишком малые L,M приводят к простым границам - первый столбец. Компромисс - второй столбец, присутствуют ошибки. Большие величины для обоих параметров - правый столбец, хороший результат, но возможно переобучение. 

# Отбор информативных признаков
## Генетический алгоритм

Информативные признаки - те, что наиболее сильно влияют на результаты обучаемой модели. Используя только оные,т.е, произведя отбор признаков и уменьшение их в исходном датасете, мы можем эффективно уменшить признаковое пространство, избавившись от ненужных данных, не теряя качества классификации, например, или не увеличивая ошибку в задачах регрессии. 

Генетический алгоритм начинается со стартовой популяции, которая состоит из набора хромосом (векторов - решений), где каждая хромосома имеет последовательность генов.

Ген в GA - строительный блок хромосомы. Для нашей задачи - гены и есть признаки. Задачи состоит в отборе оных. В используемом датасете находится 360 признаков, соответственно - 360 генов в хромосоме. Представление гена выберем бинарное (присутствует в хромосоме или нет). 

![](dia.png)

Используя функцию приспособленности, GA выбирает лучшие решения в качестве родителей для создания новой популяции, а "плохие" - "убивает". Новые решения в новой популяции создаются применением двух операций над родителями - кроссовер и мутация. 

Начальную популяцию (их хромосомное представление) мы инициализируем случайно. Это легко делается с помощью NumPy. 

![](pop.png)

Критерий для выбора родителей - значение приспособленности. Чем выше значение - тем лучше решение. Оно расчитывается с помощью функции приспособленности. Выбранная исходная модель - SVC - следовательно, мы отбираем признаки для улучшения точности классификации. Она же и будет критерием приспособленности того или иного решения. SVC будет обучаться на всех элементах обучающей выборки ТОЛЬКО с определёнными признаками.

#### Кроссовер и мутация
Опираясь на значение ФП, фильтруем решения, выбираем лучшие. GA предполагает, что совмещение двух хороших решений даст новое лучшее. Здесь и применяется кроссовер - обмен родителей определёнными генами. Был использован одноточечный кроссовер, в котором определённая точка разделяет хромосому. Гены перед оной берутся из первого решения, после - из второго. Новая популяция для новой итерации состоит из предыдущих родителей и их потомства. 
![](OnePointCrossover.svg)

Мутации в данной реализации - случайное обращение некоторых генов.

### Реализация

Извлекаем данные из запакованного в pickle части датасета Fruits360 (содержит 2000 объектов с 360 признаками, 8 классов)

```python
f = open("dataset_features.pkl", "rb")
data_inputs = pickle.load(f)
f.close()

f = open("outputs.pkl", "rb")
data_outputs = pickle.load(f)
f.close()

num_samples = data_inputs.shape[0]
num_feature_elements = data_inputs.shape[1]
train_indices = numpy.arange(1, num_samples, 4)
test_indices = numpy.arange(0, num_samples, 4)
print("Number of training samples: ", train_indices.shape[0])
print("Number of test samples: ", test_indices.shape[0])

"""
Параметры GA:
    размер популяции
    размер пула для "спаривания"
    количество мутаций
"""

sol_per_pop = 8 # размер популяции
num_parents_mating = 4 # Количество родителей в пуле.
num_mutations = 3
# Определяем размеры популяции.
pop_shape = (sol_per_pop, num_feature_elements)
 
# Создание начальной популяции.
new_population = numpy.random.randint(low=0, high=2, size=pop_shape)
print(new_population.shape)
 
best_outputs = []
num_generations = 100 #количество итераций (поколений)
```

```python
for generation in range(num_generations):
    print("Generation : ", generation)
    # Рассчитываем приспособленность для каждой хромосомы в популяции
    fitness = GA.cal_pop_fitness(new_population, data_inputs, data_outputs, train_indices, test_indices)

    best_outputs.append(numpy.max(fitness))
    # Точность классификации лучшего решения данной популяции.
    print("Best result : ", best_outputs[-1])

    # Выбор лучших родителей.
    parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)

    # Генерация следующего поколения с помощью кроссовера.
    offspring_crossover = GA.crossover(parents, offspring_size=(pop_shape[0]-parents.shape[0], num_feature_elements))

    # Применение мутаций.
    offspring_mutation = GA.mutation(offspring_crossover, num_mutations=num_mutations)

    # Создание новой популяции, родители + потомство.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
```
#### Методы в GA.py

```python
#Обучаем SVC на отобранных признаках
def cal_pop_fitness(pop, features, labels, train_indices, test_indices):
    accuracies = numpy.zeros(pop.shape[0])
    idx = 0

    for curr_solution in pop:
        reduced_features = reduce_features(curr_solution, features)
        train_data = reduced_features[train_indices, :]
        test_data = reduced_features[test_indices, :]

        train_labels = labels[train_indices]
        test_labels = labels[test_indices]

        SV_classifier = sklearn.svm.SVC(gamma='scale')
        SV_classifier.fit(X=train_data, y=train_labels)

        predictions = SV_classifier.predict(test_data)
        accuracies[idx] = classification_accuracy(test_labels, predictions)
        idx = idx + 1
    return accuracies

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # Точка для кроссовера и разделения генов, обычно - центр
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Индексы родителей
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # Первую половину генов получаем от первого родителя, вторую - от второго
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover, num_mutations=2):
    mutation_idx = numpy.random.randint(low=0, high=offspring_crossover.shape[1], size=num_mutations)
    # Мутации случайно свапают значения генов
    for idx in range(offspring_crossover.shape[0]):
        offspring_crossover[idx, mutation_idx] = 1 - offspring_crossover[idx, mutation_idx]
    return offspring_crossover
```


### Результаты


![](gen.png)
![](g2.png)

Selected indices :  [  1   6   7   8   9  11  13  14  15  16  18  19  20  24  25  27  31  32
  33  35  38  39  43  44  47  48  52  53  54  59  60  62  63  64  65  66
  68  69  73  76  82  86  88  89  91  94  97 101 103 104 105 107 108 109
 115 116 118 122 123 127 129 133 136 139 140 141 143 145 147 148 152 153
 157 159 164 165 167 168 170 172 173 174 175 176 182 186 191 193 194 197
 198 201 204 205 206 208 209 211 212 216 217 218 219 220 221 222 226 229
 230 231 232 237 240 241 244 245 248 249 255 256 260 261 262 263 268 269
 271 272 273 276 279 283 284 287 288 293 294 297 298 304 305 306 311 312
 314 318 320 321 322 325 327 328 329 331 335 338 339 340 341 342 343 344
 346 347 348 350 354 356 358 359]
 
Number of selected elements :  169

Best solution fitness :  0.9595959595959596


# Решающие деревья

Главная задача решающих деревьев состоит в классификации данных и, соответственно, аппроксимации заданной булевой функции. Признаки объектов выборки являются параметрами той самой частично заданной исходной функции *f*. Решающее дерево содержит метки:

* в узлах, не являющихся листьями: признаки, по которым различаются объекты
* В листьях: значения целевой функции
- На рёбрах: значения признака, из которого исходит ребро


Чтобы классифицировать новый объект, нужно спуститься по дереву до листа и выдать соответствующее значение

### Алгоритм построения

Выбираем очередной признак *Q*, помещаем его в корень

Для всех его значений *i*:

        Оставляем из тестовых объектов только те, у которых значение *Q = i*
        
        Рекурсивно строим дерево на этом "потомке"
    
### Энтропия

Предположим, что имеется множество *A* из *n* элементов, *m* из которых обладают некоторым свойством *S*. Тогда энтропия множества *A* по отношению к свойству *S* - это

![](https://latex.codecogs.com/svg.latex?H%28A%2CS%29%20%3D%20-%5Cfrac%7Bm%7D%7Bn%7D%5Clog_2%5Cfrac%7Bm%7D%7Bn%7D-%5Cfrac%7Bn-m%7D%7Bn%7D%5Clog_2%5Cfrac%7Bn-m%7D%7Bn%7D)

Энтропия зависит от пропорции, в котором разделяется множество. Чем "ровнее" поделили, тем больше энтропия.

Если свойство *S* - не бинарное, а может принимать *s* различных значений, каждое из которых реализуется в ![](https://latex.codecogs.com/svg.latex?m_i) случаях, то

![](https://latex.codecogs.com/svg.latex?H%28A%2C%20S%29%3D-%5Csum%20_%7B%7Bi%3D1%7D%7D%5E%7Bs%7D%5Cfrac%7Bm_%7Bi%7D%7D%7Bn%7D%5Clog%5Cfrac%7Bm_%7Bi%7D%7D%7Bn%7D)

Энтропия - среднее количество битов, которые требуютяс для кодировки признака *S* у элемента множества *A*.

### Прирост информации

Признак для  классификации нужно выбирать так, чтобы после классификации энтропия (относительно целевой функции) стала как можно меньше.

Предположим, что множество *A* элементов, характеризующихся свойством *S*, классифицировано с помощью признака *Q*, имеющего *q* значений. Тогда прирост информации (information gain) определяется:

![](https://latex.codecogs.com/svg.latex?Gain%28A%2C%20Q%29%20%3D%20H%28A%2C%20S%29%3D-%5Csum%20_%7B%7Bi%3D1%7D%7D%5E%7Bq%7D%5Cfrac%7B%7CA_%7Bi%7D%7C%7D%7B%7CA%7C%7DH%28A_i%2C%20S%29)

где ![](https://latex.codecogs.com/svg.latex?A_i) - множество элементов *A*, на которых признак *Q* имеет значение *i*

## Собственно, ID3

![](https://latex.codecogs.com/svg.latex?ID3%5Cmathcal%7B%28A%2CS%2CQ%29%7D)
1. Создать корень дерева
2. Если *S* выполняется на всех элементах *A*, поставить в корень метку 1 и выйти.
    * Если *S* не выполняется ни на одном эл-те *A*, поставить в корень метку 0 и выйти. 
    * Если *Q = 0*, то :
        * Если *S* выполняется на половине или большей части *A*, поставить в корень метку 1 и выйти.
        * Иначе, поставить в корень метку 0 и выйти.
3. Выбрать ![](https://latex.codecogs.com/svg.latex?Q%20%5Cin%20%5Cmathcal%7BQ%7D), для которого ![](https://latex.codecogs.com/svg.latex?Gain%28A%2C%20Q%29) максимален
4. Поставить в корень метку *Q*
5. Для каждого значения *q* признака *Q*:
    * Добавить нового потомка корня и пометить соответствующее исходящее ребро меткой *q*
    * Если в *A* нет элементов, где *Q = q* ![](https://latex.codecogs.com/svg.latex?%7CA_q%7C%20%3D%200), то пометить этого потомка в зависимости от того, на какой части *A* выполняется *S* (аналогично п.1)
    * Иначе запустить ![](https://latex.codecogs.com/svg.latex?ID3%28%5Cmathcal%7BA%7D_%7Bq%7D%2CS%2C%20%5Cmathcal%7BQ%7D%20%5Ctextbackslash%20Q%20%29) и добавить его результат как поддерево с корнем в этом потомке
    
### Реализация

Был взят датасет German Credit, 1000 элементов, признаков 20 (7 numerical, 13 categorical)

Attribute 1:  (qualitative)
	       Status of existing checking account
               A11 :      ... <    0 DM
	       A12 : 0 <= ... <  200 DM
	       A13 :      ... >= 200 DM /
		     salary assignments for at least 1 year
               A14 : no checking account

Attribute 2:  (numerical)
	      Duration in month

Attribute 3:  (qualitative)
	      Credit history
	      A30 : no credits taken/
		    all credits paid back duly
              A31 : all credits at this bank paid back duly
	      A32 : existing credits paid back duly till now
              A33 : delay in paying off in the past
	      A34 : critical account/
		    other credits existing (not at this bank)

Attribute 4:  (qualitative)
	      Purpose
	      A40 : car (new)
	      A41 : car (used)
	      A42 : furniture/equipment
	      A43 : radio/television
	      A44 : domestic appliances
	      A45 : repairs
	      A46 : education
	      A47 : (vacation - does not exist?)
	      A48 : retraining
	      A49 : business
	      A410 : others

|                                     |                   |                |         |               |                       |                          |                                                     |                         |                            |                         |          |              |                         |         |                                         |      |                                                          |           |                |        | 
|-------------------------------------|-------------------|----------------|---------|---------------|-----------------------|--------------------------|-----------------------------------------------------|-------------------------|----------------------------|-------------------------|----------|--------------|-------------------------|---------|-----------------------------------------|------|----------------------------------------------------------|-----------|----------------|--------| 
| Status of existing checking account | Duration in month | Credit history | Purpose | Credit amount | Savings account/bonds | Present employment since | Installment rate in percentage of disposable income | Personal status and sex | Other debtors / guarantors | Present residence since | Property | Age in years | Other installment plans | Housing | Number of existing credits at this bank | Job  | Number of people being liable to provide maintenance for | Telephone | foreign worker | target | 
| A11                                 | 6                 | A34            | A43     | 1169          | A65                   | A75                      | 4                                                   | A93                     | A101                       | 4                       | A121     | 67           | A143                    | A152    | 2                                       | A173 | 1                                                        | A192      | A201           | 1      | 
| A12                                 | 48                | A32            | A43     | 5951          | A61                   | A73                      | 2                                                   | A92                     | A101                       | 2                       | A121     | 22           | A143                    | A152    | 1                                       | A173 | 1                                                        | A191      | A201           | 2      | 
| A14                                 | 12                | A34            | A46     | 2096          | A61                   | A74                      | 2                                                   | A93                     | A101                       | 3                       | A121     | 49           | A143                    | A152    | 1                                       | A172 | 2                                                        | A191      | A201           | 1      | 
| A11                                 | 42                | A32            | A42     | 7882          | A61                   | A74                      | 2                                                   | A93                     | A103                       | 4                       | A122     |


Узлом "победителем" будет оный с максимальным приростом информации (IGain), повторяем процесс для поиска признака по которому мы будем разбивать данные в узлах. Функция ниже:
```python
def ID3(data):
	splitting_attribute = None
	split_point = None
	max_gain = 0.0
	info_d = globalFunc.calculate_info_d(data['target'])
	
	for attribute in data.columns:
		if attribute != 'target' and len(data[attribute].unique()) > 1:
			gain, temp_split_point = globalFunc.calculate_information_gain(data, attribute, info_d)
			if gain > max_gain:
				max_gain = gain
				splitting_attribute = attribute
				split_point = temp_split_point
	return splitting_attribute, split_point
```
Небольшой препроцессинг данных
```python
def preprocess(data):
	for column in data.columns:
		if len(data[column].unique()) <= 13:
			data[column] = data[column].astype(object)
	equiv = {1: 'good', 2: 'bad'}
	data['target'] = data['target'].map(equiv)
	return data
```
Считаем IGain

```python
def calculate_information_gain(data, attribute, info_d):
	attr_type = str(data[attribute].dtype)
	if attr_type.find('int') != -1 or attr_type.find('float') != -1:
		info_attribute_d, split_point = info_d_for_continuous_attribute(data, attribute)
		gain = info_d - info_attribute_d
		return gain, split_point
	else:
		info_attribute_d = info_d_for_nominal_attributes(data,attribute)
		gain = info_d - info_attribute_d
		return gain,None
```

Метод генерации самого дерева
```python
## Присваиваем лучший разбиваемый атрибут текущему узлу
root.data = splitting_attribute
if split_point == None:
	values = data[splitting_attribute].unique()

	## разбиваем данные на все возможные значения разделяемого атрибута и рекурсивно индуцируем дерево 
	for value in values:
		root.link_name.append(value)
		root.link.append(self.create_tree(data[data[splitting_attribute] == value].drop([splitting_attribute],1),algo))

else:
	root.split_point = split_point
	root.link_name.append(' A <=' + str(split_point))
	root.link.append(self.create_tree(data[data[splitting_attribute] <= split_point].drop([splitting_attribute],1),algo))
	root.link_name.append('A > '+ str(split_point))
	root.link.append(self.create_tree(data[data[splitting_attribute] >  split_point].drop([splitting_attribute],1),algo))
return root

```
Полученное дерево при фракции обучающих данных в 60%, визуализация thru GraphViz

![](ID3-0.6.png)


## C4.5, the Great and the Mighty

C4.5 является усовершенствованной версией алгоритма ID3 того же автора. В частности, в новую версию были добавлены отсечение ветвей (pruning), возможность работы с числовыми признаками, а также возможность построения дерева из неполной обучающей выборки, в которой отсутствуют значения некоторых признаками.

#### Ограничения, требования к данным:

1. **Описание признаков.** Данные в виде плоской таблицы. Объекты - конечный набор признаков, которые имеют дискретное или числовое значение. Количество признаков должно быть фиксированным для всех объектов.
2. **Определенные классы.** Каждый объект должен быть ассоциирован с конкретным классом, т.е. один из признаков должен быть выбран в качестве метки класса.

Сам алгоритм:
Алгоритм строит деревья аналогично ID3, используя концепцию энтропии. Обучающая выборка - набор ![](https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20S%3D%7Bs_%7B1%7D%2Cs_%7B2%7D%2C...%7D%7D) уже классифицированых объектов. Каждый объект ![](https://latex.codecogs.com/gif.latex?%7B%5Cdisplaystyle%20s_%7Bi%7D%7D) состоит из p-мерного вектора ![](https://latex.codecogs.com/gif.latex?%28x_%7B%7B1%2Ci%7D%7D%2Cx_%7B%7B2%2Ci%7D%7D%2C...%2Cx_%7B%7Bp%2Ci%7D%7D%29) признаков.

В каждом узле дерева алгоритм выбирает признак данных, который наиболее эффективно разбивает множество на подмножества наполненные одним классом или другим. Критерий разбиения - нормализованный прирост информации (IGain). Как и ранее, признак с наибольшим IG выбирается для решений. Алгоритм далее рекурсирует по разбитым подмножествам.


### Реализация

Собственно, для анализа и сравнения двух алгоритмов использовался один и тот же датасет.
Сам метод алгоритма C4.5:

```python
def C45(data):
	splitting_attribute = None
	max_gain_ratio = 0.0
	split_point = None
	info_d = globalFunc.calculate_info_d(data['target'])
	for attribute in data.columns:
		if attribute != 'target' and len(data[attribute].unique()) > 1:
			gain_ratio, temp_split_point = globalFunc.calculate_gain_ratio(data, attribute, info_d)
			if gain_ratio > max_gain_ratio:
				max_gain_ratio = gain_ratio
				splitting_attribute = attribute
				split_point = temp_split_point
	return splitting_attribute, split_point
```
Считаем процент IGain
```python
def calculate_gain_ratio(data, attribute, info_d):
	gain, split_point = calculate_information_gain(data, attribute, info_d)
	split_info = calculate_split_info(data, attribute)
	gain_ratio = gain / split_info
	return gain_ratio, split_point
```
Сам общий эстимейтор
```python
def estimate(row,dtree):
	if len(dtree.link) == 0:
		return dtree.data
	for (ln,l) in zip(dtree.link_name,dtree.link):
		if dtree.split_point != None:
			if row[dtree.data] <= dtree.split_point and str(ln).find('<') != -1:
				return estimate(row,l)
			elif row[dtree.data] > dtree.split_point and str(ln).find('>') != -1:
				return estimate(row,l)
		elif row[dtree.data] == ln:
			return estimate(row,l)
```   
Основной метод (почти)

#### Результат

Визуализация аналогична ID3, деревья получились большие...

![](C4.5-0.6.png)

### Сравнение
[Сравнение тут, спасибо GithubPages за хост](https://zoncker.github.io/ML2/ "asd")


## Метод поиска в глубину и ширину

### DFS(Branch and bound)

1. Нумерация признаков по возрастанию номеров —
чтобы избежать повторов при переборе подмножеств
2. если набор *J* бесперспективен, то больше не пытаться его наращивать.

![](https://latex.codecogs.com/svg.latex?Q%5E*_j) - значение критерия на самом лучшем наборе мощности *j* из всех до сих пор просмотренных.

Оценка бесперспективности набора признаков *J*: набор J не наращивается, если

![](https://latex.codecogs.com/svg.latex?%5Cexists%20j%3A%20Q%28J%29%5Cgeq%5Cmu%20Q%5E*_j) и ![](https://latex.codecogs.com/svg.latex?%7CJ%7C%20%5Cgeq%20j%20&plus;%20d)

![](https://latex.codecogs.com/svg.latex?d%20%5Cgeq%200) - целочисленный параметр, 

![](https://latex.codecogs.com/svg.latex?%5Cmu%20%5Cgeq%201) - вещественный параметр.

Чем меньше ![](https://latex.codecogs.com/svg.latex?d) и ![](https://latex.codecogs.com/svg.latex?%5Cmu), тем сильнее сокращается перебор.

Вход: множество ![](https://latex.codecogs.com/svg.latex?F), критерий ![](https://latex.codecogs.com/svg.latex?Q), параметры ![](https://latex.codecogs.com/svg.latex?d%2C%20%5Cmu)

Нарастить (J)

**если** найдётся ![](https://latex.codecogs.com/svg.latex?j%20%5Cleq%20%7Cd%7C%3A%20Q%28J%29%20%5Cgeq%20%5Cmu%20Q%5E*_j), то 

выход

![](https://latex.codecogs.com/svg.latex?Q%5E*_%7B%7CJ%7C%7D%3A%3D%5Cmin%5Cleft%20%5C%7B%20Q%5E*_%7B%7CJ%7C%7D%2CQ%28J%29%20%5Cright%20%5C%7D)

![](https://latex.codecogs.com/svg.latex?%5Cforall%20f_s%20%5Cin%20F%3A%20s%3E%20%5Cmax%20%5Cleft%20%5C%7B%20t%7C%20f_t%20%5Cin%20J%20%5Cright%20%5C%7D)

Инициализация массива лучших значений критерия:

![](https://latex.codecogs.com/svg.latex?Q_j%5E*%3A%3DQ%28%5Cvarnothing%20%29%20%5Cforall%20j%20%3D%201%2C%5Cdots%2Cn)

Упорядочить признаки по убыванию информативности

Нарастить ![](https://latex.codecogs.com/svg.latex?%28%5Cvarnothing%20%29)

Вернуть *J*, для которого ![](https://latex.codecogs.com/svg.latex?Q%28J%29%3D%20%5Cmin_%7Bj%3D1%2C%5Cdots%2C%20n%7D%20Q_j%5E*)

#### Реализация

Всё те же ирисы, критерий - CV конкретного признака.
```python
def criterion_function(features):

    X_train, X_test, y_train, y_test = train_test_split(iris.data[:, features],
                                                        iris.target, test_size=0.4, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    result = cross_val_score(knn, iris.data[:, features], iris.target, cv=5)
    return max(result)
```
Основной метод алгоритма:
```python
def branch_and_bound(root, D, d):
    global flag
    global J_max
    global result_node

    # Начальное вычисление критерия в корне
    root.J = criterion_function(root.features)

    # Не наращивать набор (ветвиться от этого узла) если J <= J_max
    if flag == False and root.J <= J_max:
        return

    #Если это лист, обновить J_max, result_node и выйти
    if root.level == D - d:
        if flag == True:
            J_max = root.J
            flag = False
            result_node = root

        elif root.J > J_max:
            J_max = root.J
            result_node = root

        return
```
Считаем количество ветвей этого узла
```python
    no_of_branches = (d + 1) - len(root.preserved_features)

    # Генерим эти ветви
    branch_feature_values = sorted(
        random.sample(list(set(root.features) - set(root.preserved_features)), no_of_branches))

    #Итерируем по ветвям, и для каждой ветви рекурсивно коллим метод
    for i, branch_value in enumerate(branch_feature_values):
        child = tree_node(branch_value, [value for value in root.features if value != branch_value], \
                          root.preserved_features + branch_feature_values[i + 1:], root.level + 1)

        root.children.append(child)

        branch_and_bound(child, D, d)
```
Собственно, отрисовка дерева
```python
def display_tree(node, dot_object, parent_index):
    iris = datasets.load_iris()
    for i in node.features:
        node.name.append(iris.feature_names[i])

    #Создать узел в дереве
    dot_object.node(str(node.index),
                    "Features = " + str(node.name) + "\nJ(Features) = " + str(node.J) + "\nPreserved = " + str(
                        node.preserved_features))

    #Если не корень, то создать ветвь к родителю
    if node.index != -1:
        dot_object.edge(str(parent_index), str(node.index), label=str(node.branch_value))

    # базовый случай, когда у узла нет потомков, выйти 
    if len(node.children) == 0:
        return

    # рекурсивный колл метода для всех потомков этого узла
    for child in reversed(node.children):
        display_tree(child, dot_object, node.index)
```

![](bb_tree.png)

### BFS

Он же многорядный итерационный алгоритм МГУА (МГУА — метод группового учёта аргументов)

Философия — принцип неокончательных решений Габора:

принимая решения, следует оставлять максимальную свободу выбора для принятия последующих решений.

Усовершенствуем алгоритм Add:

на каждой *j*-й итерации будем строить не один набор, а множество из ![](https://latex.codecogs.com/svg.latex?B_j) наборов, называемое *j*-м рядом:

![](https://latex.codecogs.com/svg.latex?R_j%20%3D%20%5Cleft%20%5C%7B%20J_j%5E1%2C%5Cdots%2CJ_j%5E%7BB_j%7D%20%5Cright%20%5C%7D%2C%20J_j%5Eb%20%5Csubseteq%20F%2C%20%7CJ%5Eb_j%7C%3Dj%2C%20b%20%3D%201%2C%5Cdots%2CB_j)

где ![](https://latex.codecogs.com/svg.latex?B_j%20%5Cleq%20B)

Вход: множество *F*, критерий *Q*, параметры *d*, *B*

первый ряд состоит из всех наборов длины 1:

![](https://latex.codecogs.com/svg.latex?R_1%3A%3D%5Cleft%20%5C%7B%20%5Cleft%20%5C%7B%20f_1%20%5Cright%20%5C%7D%2C%5Cdots%2C%5Cleft%20%5C%7B%20f_n%20%5Cright%20%5C%7D%20%5Cright%20%5C%7D%3B%20Q%5E*%20%3D%20Q%28%5Cvarnothing%20%29%3B)

![](https://latex.codecogs.com/svg.latex?%5Cforall%20j%3D1%2C%5Cdots%2Cn) где j - сложность наборов:

отсортировать ряд ![](https://latex.codecogs.com/svg.latex?R_j%20%3D%20%5Cleft%20%5C%7B%20J_j%5E1%2C%5Cdots%2CJ_j%5E%7BB_j%7D%20%5Cright%20%5C%7D)

по возрастанию критерия: ![](https://latex.codecogs.com/svg.latex?Q%28J_j%5E1%29%5Cleq%20%5Cdots%20%5Cleq%20Q%28J_j%5E%7BB_j%7D%29%3A)

![](https://latex.codecogs.com/svg.latex?if%20B_j%20%3E%20B%20%5Crightarrow)

![](https://latex.codecogs.com/svg.latex?R_j%3A%3D%5Cleft%20%5C%7B%20J_j%5E1%2C%5Cdots%2CJ_j%5EB%20%5Cright%20%5C%7D) - *B* лучших наборов ряда;

![](https://latex.codecogs.com/svg.latex?if%20Q%28J_j%5E1%29%3CQ%5E*%20%5Crightarrow%20j%5E*%3A%3Dj%3B%20Q%5E*%3A%3DQ%28J_j%5E1%29%3B)

![](https://latex.codecogs.com/svg.latex?if%20j-j%5E*%20%5Cgeq%20d%20%5Crightarrow%20J_%7Bj%5E*%7D%5E1%3B)

породить следующий ряд:

![](https://latex.codecogs.com/svg.latex?R_%7Bj&plus;1%7D%3A%3D%20%5Cleft%20%5C%7B%20J%5Ccup%20%5Cleft%20%5C%7B%20f%20%5Cright%20%5C%7D%20%7C%20J%20%5Cin%20R_j%2C%20f%20%5Cin%20FJ%20%5Cright%20%5C%7D%3B)

Трудоёмкость:

![](https://latex.codecogs.com/svg.latex?O%28Bn%5E2%29) точнее ![](https://latex.codecogs.com/svg.latex?O%28Bn%28j%5E*&plus;d%29%29)

Проблема дубликатов:

после сортировки (шаг 3) проверить на совпадение только соседние наборы с равными значениями внутреннего и внешнего критерия.

Адаптивный отбор признаков:

на шаге 8 добавлять к j-му ряду только признаки *f* с наибольшей информативностью  ![](https://latex.codecogs.com/svg.latex?l_j%28f%29%3A)

![](https://latex.codecogs.com/svg.latex?l_j%28f%29%3D%5Csum_%7Bb%3D1%7D%5E%7BB_j%7D%5Cleft%20%5B%20f%20%5Cin%20%5Cright%20J_j%5Eb%20%5D)


### Exhaustive Feature Selection

Вход: множество *F*, критерий *Q*, параметры *d*

![](https://latex.codecogs.com/svg.latex?Q%5E*%3A%3DQ%28%20%5Cvarnothing%20%29) - инициализация

![](https://latex.codecogs.com/svg.latex?%5Cforall%20j%20%3D%201%2C%5Cdots%2Cn), где *j* - сложность наборов

найти лучший набор сложности *j*:

![](https://latex.codecogs.com/svg.latex?J_j%3A%3D%5Carg%5Cmin_%7BJ%3A%7CJ%7C%3Dj%7D%20Q%28J%29)

if ![](https://latex.codecogs.com/svg.latex?Q%28J_j%29%20%3C%20Q%5E*%5Crightarrow%20j%5E*%3A%3Dj%3B) ![](https://latex.codecogs.com/svg.latex?Q%5E*%3A%3DQ%28J_j%29)

if ![](https://latex.codecogs.com/svg.latex?j-j%5E*%20%5Cgeq%20d) то вернуть ![](https://latex.codecogs.com/svg.latex?J_%7Bj%5E*%7D)

Преимущества
- простота реализации
- гарантированный результат
- полный перебор эффективен когда
  - информативных признаков не много, ![](https://latex.codecogs.com/svg.latex?j%5E*%20%3C_%5Csim%205)
  - всего признаков не много, ![](https://latex.codecogs.com/svg.latex?n%20%3C_%5Csim%2020..100)
  
Недостатки:
- В остальных случаях ооочень долго - ![](https://latex.codecogs.com/svg.latex?O%282%5En%29)

- чем больше перебирается вариантов, тем больше переобучение (особенно, если лучшие из вариантов существенно различны и одинаково плохи)

### Реализация

***Irises, irises never change.***

Собсна, нашим критерий - скоринг точности CV на KNN.
```python
def _calc_score(selector, X, y, indices, **fit_params):
    if selector.cv:
        scores = cross_val_score(selector.est_,
                                 X[:, indices], y,
                                 cv=selector.cv,
                                 scoring=selector.scorer,
                                 n_jobs=1,
                                 pre_dispatch=selector.pre_dispatch,
                                 fit_params=fit_params)
    else:
        selector.est_.fit(X[:, indices], y, **fit_params)
        scores = np.array([selector.scorer(selector.est_, X[:, indices], y)])
    return indices, scores
```
Количество комбинаций признаков длины r из n.
```python
r = min(r, n-r)
if r == 0:
return 1
numer = reduce(op.mul, range(n, n-r, -1))
denom = reduce(op.mul, range(1, r+1))
return numer//denom

all_comb = np.sum([ncr(n=X_.shape[1], r=i)
		   for i in range(self.min_features,
				  self.max_features + 1)])
```
Производим отбор признаков и обучение модели на выборке
```python
candidates = chain(*((combinations(range(X_.shape[1]), r=i))
		   for i in range(self.min_features,
				  self.max_features + 1)))
self.subsets_[iteration] = {'feature_idx': c,
			    'cv_scores': cv_scores,
			    'avg_score': np.mean(cv_scores)}

max_score = float('-inf')
for c in self.subsets_:
    if self.subsets_[c]['avg_score'] > max_score:
	max_score = self.subsets_[c]['avg_score']
	best_subset = c
score = max_score
idx = self.subsets_[best_subset]['feature_idx']

self.best_idx_ = idx
self.best_score_ = score
self.fitted = True
self.subsets_, self.best_feature_names_ = \
    _get_featurenames(self.subsets_,
		      self.best_idx_,
		      custom_feature_names,
		      X)
```
Получаем словарь где каждое значение - список с кол-вом итераций (кол-во подвыборок признаков) как его длины. Ключи словаря соответствуют спискам.
```python
    def get_metric_dict(self, confidence_interval=0.95):
        fdict = deepcopy(self.subsets_)
        for k in fdict:
            std_dev = np.std(self.subsets_[k]['cv_scores'])
            bound, std_err = self._calc_confidence(
                self.subsets_[k]['cv_scores'],
                confidence=confidence_interval)
            fdict[k]['ci_bound'] = bound
            fdict[k]['std_dev'] = std_dev
            fdict[k]['std_err'] = std_err
        return fdict

    def _calc_confidence(self, ary, confidence=0.95):
        std_err = scipy.stats.sem(ary)
        bound = std_err * sp.stats.t._ppf((1 + confidence) / 2.0, len(ary))
        return bound, std_err
```

![](efs.png)

Best accuracy score: 0.97
Best subset (indices): (0, 2, 3)
Best subset (corresponding names): ('sepal length', 'petal length', 'petal width')


## Жадное добавление - Add

- работает быстро ![](https://latex.codecogs.com/svg.latex?O%28n%5E2%29) точнее - ![](https://latex.codecogs.com/svg.latex?O%28n%28j%5E*&plus;d%29%29)
- возможны быстрые инкрементные алгоритмы,
Недостатки:
- Add склонен включать в набор лишние признаки.
Способы устранения:
- Add-Del — чередование добавлений и удалений (см. далее);
- поиск в ширину (см. выше)

![](https://latex.codecogs.com/svg.latex?J_0%3A%3D%5Cvarnothing%20%3B%20Q%5E*%3A%3DQ%28%5Cvarnothing%29%3B%20t%3A%3D0%3B) - инициализация


повторять


пока ![](https://latex.codecogs.com/svg.latex?%5Cleft%20%7CJ_t%20%5Cright%20%7C%3Cn) добавлять признаки


![](https://latex.codecogs.com/svg.latex?t%3A%3Dt&plus;1%3B) - next iter


![](https://latex.codecogs.com/svg.latex?F*%3A%3D%5Carg%5Cmin_%7Bf%20%5Cin%20F%20%5Csetminus%20J_%7Bt-1%7D%20%7D%20Q%28J_%7Bt-1%7D%5Ccup%20%5Cleft%20%5C%7B%20f%20%5Cright%20%5C%7D%29%3B%20J_t%3A%3DJ_%7Bt-1%7D%5Ccup%20%5Cleft%20%5C%7B%20f%5E*%20%5Cright%20%5C%7D%3B) - наиболее выгодный признак добавляем в набор;


если ![](https://latex.codecogs.com/svg.latex?Q%28J_j%29%20%3CQ%5E*%20%5Crightarrow%20j%5E*%3A%3Dj%3B%20Q%5E*%3A%3DQ%28J_j%29)


если ![](https://latex.codecogs.com/svg.latex?j-j%5E*%20%5Cgeq%20d%20%5Crightarrow%20J_%7Bj%5E*%7D)
	
Best accuracy score: 0.973
Best subset (corresponding names): ('sepal length', 'petal length', 'petal width')


## удаление - Del


![](https://latex.codecogs.com/svg.latex?J_0%3A%3D%5Cvarnothing%20%3B%20Q%5E*%3A%3DQ%28%5Cvarnothing%29%3B%20t%3A%3D0%3B) - инициализация


повторять


пока ![](https://latex.codecogs.com/svg.latex?%5Cleft%20%7C%20J_t%20%5Cright%20%7C%3E0) удалять признаки


![](https://latex.codecogs.com/svg.latex?t%3A%3Dt&plus;1%3B) - next iter


![](https://latex.codecogs.com/svg.latex?F*%3A%3D%5Carg%5Cmin_%7Bf%20%5Cin%20F%20%5Csetminus%20J_%7Bt-1%7D%20%7D%20Q%28J_%7Bt-1%7D%5Ccup%20%5Cleft%20%5C%7B%20f%20%5Cright%20%5C%7D%29%3B%20J_t%3A%3DJ_%7Bt-1%7D%5Ccup%20%5Cleft%20%5C%7B%20f%5E*%20%5Cright%20%5C%7D%3B) - наиболее выгодный признак добавляем в набор;


если ![](https://latex.codecogs.com/svg.latex?Q%28J_j%29%20%3CQ%5E*%20%5Crightarrow%20j%5E*%3A%3Dj%3B%20Q%5E*%3A%3DQ%28J_j%29)


если ![](https://latex.codecogs.com/svg.latex?j-j%5E*%20%5Cgeq%20d%20%5Crightarrow%20J_%7Bj%5E*%7D)

Del
Best accuracy score: 0.973
Best subset (corresponding names): ('sepal length', 'petal length', 'petal width')



Add-Del
Best accuracy score: 0.973
Best subset (corresponding names): ('sepal length', 'petal length', 'petal width')

