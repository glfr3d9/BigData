import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, fpmax, association_rules
from mlxtend.preprocessing import TransactionEncoder

print('2. Загрузить данные в датафрейм')
all_data = pd.read_csv('groceries - groceries.csv')
print(all_data)  # Видно, что датафрейм содержит NaN значения
print('3. Переформируем данные удалив все значения NaN')
np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem, str)] for row in np_data]
print('4,5. Получим список всех уникальных товаров и их количество')
unique_items = {}
for row in np_data:
    for elem in row:
        if elem in unique_items:
            unique_items[elem] += 1
        else:
            unique_items[elem] = 1
unique_items_df = pd.DataFrame.from_dict(unique_items, orient='index', columns=['total'])
print(unique_items_df)
print('1. Преобразуем данные к виду, удобному для анализа')
te = TransactionEncoder()
te_ary = te.fit(np_data).transform(np_data)
data = pd.DataFrame(te_ary, columns=te.columns_)
print(data)
print('2. Проведем ассоциативный анализ используя алгоритм FPGrowth при уровне поддержки 0.03')
result = fpgrowth(data, min_support=0.03, use_colnames=True)
print(result)
print('3. Определите минимальное и максимальное значения для уровня поддержки для набора из 1,2, и.т.д. объектов')
for i in np.arange(0.01, 0.3, 0.01):
    result = fpgrowth(data, min_support=i, use_colnames=True)
    result['length'] = result['itemsets'].apply(lambda x: len(x))
    print(
        f'{np.round(i, decimals=2)}: 1: {1 in list(result["length"])}, 2: {2 in list(result["length"])}, 3:{3 in list(result["length"])}')
print('В рамках проведённых тестов минимальная поддержка для всех = 0.01, максимальная: 1: 0.25, 2: 0.07, 3: 0.02')
print('4. Проведите аналогичный анализ используя алгоритм FPMax')
data = pd.DataFrame(te_ary, columns=te.columns_)
result = fpmax(data, min_support=0.03, use_colnames=True)
print(result)
print('6.  Постройте гистограмму для каждого товара')
unique_items_df.reset_index(inplace=True)
unique_items_df = unique_items_df.rename(columns={'index': 'item_name'})
print(unique_items_df)
var_names = unique_items_df.columns
items_hist = unique_items_df.hist(bins=10)
plt.show()
print('7. Преобразуем набор данных, чтобы он содержал ограниченный набор товаров')
items = ['whole milk', 'yogurt', 'soda', 'tropical fruit', 'shopping bags', 'sausage', 'whipped/sour cream',
         'rolls/buns', 'other vegetables', 'root vegetables', 'pork', 'bottled water', 'pastry', 'citrus fruit',
         'canned beer', 'bottled beer']
np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem, str) and elem in items] for row in np_data]
te_ary = te.fit(np_data).transform(np_data)
data = pd.DataFrame(te_ary, columns=te.columns_)
print(data)
print('8. Проведите анализ FPGrowth и FPMax для нового набора данных. Проанализируйте, что изменилось')
result = fpgrowth(data, min_support=0.03, use_colnames=True)
print('FPGrowth\n', result)
result = fpmax(data, min_support=0.03, use_colnames=True)
print('FPMax\n', result)
print('1. Сформируем набор данных из определенных товаров и так, чтобы размер транзакции был 2 и более')
np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem, str) and elem in
            items] for row in np_data]
np_data = [row for row in np_data if len(row) > 1]
print('2. Получим частоты наборов используя алгоритм FPGrowth')
result = fpgrowth(data, min_support=0.05, use_colnames = True)
print(result)
print('3. Проведем ассоциативный анализ')
rules = association_rules(result, min_threshold = 0.3)
print(rules)
print('4. Определите, на основе какой метрики проводится расчет')
print('Расчёт производится на основе метрики confidence')
print('5. Проведите построение ассоциативных правил для различных метрик')
for i in ['confidence', 'lift', 'conviction']:
    rules = association_rules(result, metric=i, min_threshold=0.1)
    print(i)
    print(rules)
print('7. Постройте граф для следующего анализа')
rules = association_rules(result, min_threshold = 0.4, metric='confidence')
print(rules)