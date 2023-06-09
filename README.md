# TGT Digital Challenge
---
### **Описание**:
TGT Diagnostics – международная нефтесервисная компания. Системы и сервисы TGT Diagnostics предоставляют точную и надежную информацию о техническом состоянии нефтегазовых скважин. Одна из серьезных проблем, с которой сталкиваются при эксплуатации скважины, это вынос песка из нефтегазовых пластов.

  Вынос песка влияет на производительность и целостность конструкции скважин и оборудования. Для определения интервалов выноса песка мы используется скважинный шумомер, который измеряет акустические колебания в скважине. Шумомер регистрирует не только шум от движения потоков жидкости и газа в скважине, но и удары песчинок о корпус прибора.

### **Задача**:
Разработать автоматический распознаватель удара песчинок на аудиоданных.

### **Решение**:
1. **Загрузка данных и библиотек:** <br>
   Стек используемых библиотек: `NumPy`, `Pandas`, `SciPy`, `Librosa`, `Scikit-learn`, `CatBoost`, `Imbalanced-learn`, `Matplotlib`, `Seaborn`.
2. **EDA и предобработка данных:** <br>
   Реализованы следующие функции:
   - `generate_features_from_ts(data)` - используется для извлечения признаков из временного ряда.
   - `squared_std_sums(data)` - вычисляет сумму квадратов стандартных отклонений временного ряда. Функция вычисляет сумму квадратов стандартных отклонений и сумму квадратов стандартных отклонений, поделенную на длину временного ряда, и возвращает кортеж из двух значений.
   - `count_sign_reversals(data)` - подсчитывает количество переключений знака во временном ряде.
   - `extract_audio_features(data)` - подготавливает данные перед извлечением признаков из звукового файла. Эта функция вычисляет спектральную плотность мощности звукового файла и извлекает несколько признаков из этой плотности: спектральный центроид, ширину полосы пропускания, плоскость спектра, спектральную огибающую и количество пересечений нуля.
3. **Обучение CatBoost:** <br>
    На данном этапе происходит оценка качества модели машинного обучения с помощью кросс-валидации. Функция `cv()` выполняет кросс-валидацию на данных `X_` и `y_`, используя параметры модели, определенные в словаре `params`. Разбиение на фолды осуществляется с помощью класса `ShuffleSplit`. <br>
    Создается экземпляр классификатора `CatBoost` с параметрами, указанными в аргументах конструктора, и происходит обучение на данных, которые хранятся в объекте `Pool` с признаками `X_` и метками `y_`. Обучение проходит с выводом логов на каждой итерации, так как 'verbose' установлен равным 50.
