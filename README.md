<a name="readme-top"></a>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[!['Black'](https://img.shields.io/badge/code_style-black-black?style=for-the-badge)](https://github.com/psf/black)

  <h1 align="center">My first NLP project</h1>

### О проекте

На данном этапе проект включает в себя парсер новостного сайта [Лента.ру](https://lenta.ru/), первичный анализ данных, векторизацию текста статей, а также модели на основе полученных данных для решения задачи классификации.

### Задание состоит из четырех частей:

* Часть 1: изучите популярные новостные сайты (РБК, Лента и т.д.), выберите подходящий для сбора данных

* Часть 2: соберите данные удобным вам способом (по api, используя библиотеку requests или selenium)

* Часть 3: спарсите собранный массив данных таким образом, чтобы тексту каждой новости соответствовал тег (или теги) отмеченные на сайте

* Часть 4: проведите разведочный анализ данных (по аналогии с анализом в представленном ниже ноутбуке) 


**Часть 1**

Был выбран новостной сайт [Лента.ру](https://lenta.ru/) для парсинга

**Часть 2**

C помощью следующих библиотек было произведенно сбор данных с сайта:
```python
import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from IPython import display
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
```

**Часть 3**

В итоги получился следующий DF размером `(26059, 6)`:

| url | title | topic | datetime | content | subtitle |
|----------|----------|----------|-|-|-|
| url статьи | Заголовок статьи | Рубрика | Время публикации статьи| Основной текст статьи | Подзаголовок статьи| 


**Часть 4**

Был проведен разведочный анализ данных, Exploratory Data Analysis (EDA):

- Загрузка и первичный осмотр данных;
- Обработка пропущенных значений и изменение тип данных;
- Добавление новых фич;
- Изучение категориальных переменных;
- Визуализация результатов EDA
- Распределение категорий
- Распределение времени новостей
- Распределение по длине заголовка
- Распределение по длине вступительного текста
- Распределение по длине основного текста
- Ключевые слова по каждой из тематик

### Векторное представление слов и построение моделей 
Были использованы следующие бибилиотеки:
```python
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import gensim.downloader
from gensim.models import Word2Vec, KeyedVectors
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger

import mlxtend
from mlxtend.evaluate import paired_ttest_kfold_cv

from plotly.offline import iplot
import cufflinks as cf
```
#### Предобработка текста
Чтобы получить более точное и компактное представление текстов были совершены следующие этапы:

- токенизация
- лемматизация 
- сегментация
- удаление стоп-слов

С помощью **Word2Vect** были получены векторные представления слов. 
Для построения модели получили вектора предложений двумя способами:
- Усреднить вектора слов, входящих в предложение
- Взвесить вектора слов, входящих в предложение на основании их tf-idf весов

#### Построение моделей
Решили задачу классификации с помощью нескольких моделей:
- Логистическая регрессия
- Метод опорных векторов

## Лицензия

Веб-сайт [Лента.ру](https://lenta.ru/) не предоставляет информацию о конкретной лицензии, по которой публикуется его контент. Таким образом, неясно, какая лицензия применяется к контенту на этом веб-сайте.
[Политика конфиденциальности](https://lenta.ru/info/posts/privacy_policy)

