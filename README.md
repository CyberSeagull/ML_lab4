# ML_lab4
## Характеристики train та test датасетів
![image](https://github.com/CyberSeagull/ML_lab4/assets/62190177/49bbb089-e5f8-41d2-b36c-f500d43e5da8)
![image](https://github.com/CyberSeagull/ML_lab4/assets/62190177/728c6dbb-3e4f-4c6f-9c64-39b5174dc377)
![image](https://github.com/CyberSeagull/ML_lab4/assets/62190177/c5973c1f-75a0-46f1-9803-0730b70b9b08)
![image](https://github.com/CyberSeagull/ML_lab4/assets/62190177/d3331cac-c8ae-442d-ab0a-f67de6d2fdc1)
![image](https://github.com/CyberSeagull/ML_lab4/assets/62190177/1605486f-d986-46f0-8281-8935ea8d5524)
![image](https://github.com/CyberSeagull/ML_lab4/assets/62190177/ef352f1d-d593-49fb-bf5b-e4cdc74e69f1)
![image](https://github.com/CyberSeagull/ML_lab4/assets/62190177/1e3e7cea-cdec-4242-a4c9-f8a458ea709e)
![image](https://github.com/CyberSeagull/ML_lab4/assets/62190177/f47f7308-a4fc-472b-9e14-b7a75c21a214)
![image](https://github.com/CyberSeagull/ML_lab4/assets/62190177/3b7cb0ca-0298-49b5-adb7-72ddd787e81e)
## Обробка датасетів
Було вилучено спецсимволи, слова з stop_words NLTK (наприклад, y’, ‘ain’, ‘aren’, “aren’t”, ‘couldn’, “couldn’t”, ‘didn’, “didn’t”); крім цього, 
текст було переведено у нижній регістр та, зрештою, виконано процедуру стемінгу за допомогою SnowballStemmer.
## Векторизація та PCA

## Підбір моделей
## Тестування
Тестування було проведено за допомогою модуля unittest: було протестовано усі методи, написані для передобробки даних та ті, що спираються на застосування моделей 
машинного навчання (створення моделі, генерація ймовірностей та збереження результатів):


![image](https://github.com/CyberSeagull/ML_lab4/assets/62190177/93459bc0-a750-45b2-863c-2cdf9582a1fb)

Для визначення відсотку покриття тестами було застосовано coverage.py:

![image](https://github.com/CyberSeagull/ML_lab4/assets/62190177/8373a21a-12ad-4b3e-b651-27563e861409)
