# БИБЛИОТЕКИ --------------------------------------------------------------

  
# Загрузка библиотек
library(dplyr)
library(ggplot2)
library(ggpubr)
library(dplyr)
library(xgboost)
library(MLmetrics)

# Отключение предупреждений
options(warn=-1)


# ЗАГРУЗКА ДАННЫХ ---------------------------------------------------------


# Загрузка тренировочной выборки
train <- read.csv("train.csv")

# Загрузка тестовой выборки
test <- read.csv("test.csv")

# Размерность тренировочных данных
print(paste("Тренировочные данные: Количество строк: ", nrow(train), 
            "Количество столбцов: ", ncol(train)))
# Размерность тестовых данных
print(paste("Тестовые данные: Количество строк: ", nrow(test), 
            "Количество столбцов: ", ncol(test)))

train_targets <- train$label
test_targets <- test$label

# Расределение таргета
targets_str <- (
  train %>% mutate(label = ifelse(
    label == 0, 
    "Нет песка", 
    "Есть песок")
  ))$label
table(targets_str)


# РАЗВЕДОЧНЫЙ АНАЛИЗ И ПРЕДОБРАБОТКА --------------------------------------


# Функция для отображения графиков амлитуд случайных сэмплов
draw_plots <- function(have_sand = FALSE){
  raw_data <- train[-which(train$label == as.integer(!have_sand)), ]
  raw_data <- raw_data[, -which(names(raw_data) == "label")]
  
  ID <- round(runif(4, 1, nrow(raw_data)))
  plots <- list()
  names <- c("A", "B", "C", "D")
  
  for (n_plot in c(1:4)) {
    sample_data <- na.omit(as.vector(t(raw_data[ID[n_plot], ])))
    plots[[names[n_plot]]] <- local({
      name <- names[n_plot]
      ggplot(data.frame(ms = c(1:length(sample_data)), amplitude = sample_data), 
             aes(x = ms, y = amplitude)) +
        geom_line() +
        geom_smooth(method = lm) +
        geom_point() +
        labs(title = paste0("График амплитуды для наблюдения ID ", ID[1]),
             x = "Миллисекунда",
             y = "Амплитуда")
    })
  }
  
  text <- c("Случайные аудио-сэмплы без песка", 
            "Случайные аудио-сэмплы с песком")
  figure <- ggarrange(plotlist = plots,
                      labels = c("A", "B", "C", "D"),
                      ncol = 2, nrow = 2)
  annotate_figure(
    figure,
    top = text_grob(
      text[as.integer(have_sand) + 1], 
      color = "red", 
      face = "bold", 
      size = 14
    ),
    bottom = text_grob(
      "Исходные данные: \n без предобработки", 
      color = "blue",
      hjust = 1, x = 1, 
      face = "italic", 
      size = 10
    ),
  )
}

# Случайные сэмплы без песка
draw_plots(have_sand = FALSE)

# Случайные сэмплы с песком
draw_plots(have_sand = TRUE)


# Фукнция для извлечения признаков из временных рядов
generate_features_from_ts <- function(data){
  data <- na.omit(as.vector(data))
  features <- c(
    mean(data),
    sd(data),
    min(data),
    max(data),
    quantile(data, probs = 0.25),
    quantile(data, probs = 0.50),
    quantile(data, probs = 0.75)
  )
  names(features) <- c("MEAN", "STD", "MIN", "MAX", 
                       "FEATURE25", "FEATURE50", "FEATURE75")
  return(features)
}


# Функция для вычисления суммы квадратов отклонений от оси X
squared_std_sums <- function(data) {
  data <- na.omit(as.vector(data))
  zeros <- replicate(length(data), 0)
  std_sums <- sum((data - zeros) ^ 2)
  std_sums_per_len <- std_sums / length(data)
  features <- c(
    std_sums,
    std_sums_per_len
  )
  names(features) <- c("squared_std_sum", "squared_std_sum_per_len")
  return(features)
}


# Функция для БПФ
fft_func <- function(data) {
  data[is.na(data)] = 0
  fft_result <- fft(data)
  return(as.numeric(fft_result))
}


# Основная функция для предобработки данных
data_preprocessing <- function(data) {
  raw_data <- data[, -which(names(data) == "label")]
  result_data <- data.frame()
  
  # Выделяем признаки из временных рядов
  result_data <- cbind(raw_data,
                    t(apply(raw_data, MARGIN = 1, 
                            FUN = generate_features_from_ts)))
  # Считаем квадраты отклонений от оси Х
  result_data <- cbind(result_data,
                    t(apply(raw_data, MARGIN = 1, 
                            FUN = generate_features_from_ts)))
  # Добавляем колонки с результатом БПФ
  fft_df <- t(apply(raw_data, MARGIN = 1, FUN = fft_func))
  fft_df <- data.frame(fft_df) %>% select(c(1:150))
  result_data <- cbind(result_data, data.frame(fft_df))
  return(result_data)
}

# Применим основную функцию для обработки исходных данных
preprocessing_data_train <- data_preprocessing(train)
preprocessing_data_test <- data_preprocessing(test)


# Функция для отображения графиков (БПФ)
draw_fft_plots <- function(have_sand = FALSE){
  preprocessing_data <- preprocessing_data_train
  preprocessing_data$label = train_targets
  data <- preprocessing_data[-which(preprocessing_data$label == 
                                      as.integer(!have_sand)), ]
  data <- data[, -which(names(data) == "label")]
  data <- data %>% select(c(315:464))
  
  ID <- round(runif(4, 1, nrow(data)))
  
  plots <- list()
  names <- c("A", "B", "C", "D")
  
  for (n_plot in c(1:4)) {
    sample_data <- na.omit(as.vector(t(data[ID[n_plot], ])))
    plots[[names[n_plot]]] <- local({
      name <- names[n_plot]
      ggplot(data.frame(ms = c(1:length(sample_data)), 
                        amplitude = sample_data), 
             aes(x = ms, y = amplitude)) +
        geom_line() +
        geom_smooth(method = lm) +
        geom_point() +
        labs(title = paste0("График амплитуды для наблюдения ID ", ID[1]),
             x = "Частота",
             y = "Амплитуда")
    })
  }
  
  text <- c("Случайные аудио-сэмплы без песка", 
            "Случайные аудио-сэмплы с песком")
  figure <- ggarrange(plotlist = plots,
                      labels = c("A", "B", "C", "D"),
                      ncol = 2, nrow = 2)
  annotate_figure(
    figure,
    top = text_grob(
      text[as.integer(have_sand) + 1], 
      color = "red", 
      face = "bold", 
      size = 14
    ),
    bottom = text_grob(
      "Предобработанные данные:\n быстрое преобразование Фурье", 
      color = "blue",
      hjust = 1, x = 1, 
      face = "italic", 
      size = 10
    ),
  )
}

# Случайные сэмплы без песка
draw_fft_plots(have_sand = FALSE)

# Случайные сэмплы с песком
draw_fft_plots(have_sand = TRUE)


# ОБУЧЕНИЕ XGBoost --------------------------------------------------------


# Преобразуем данные к xgb.DMatrix формату
Dtrain = xgb.DMatrix( # Тренировочная матрица
  data = as.matrix(preprocessing_data_train),
  label = train_targets
)
Dtest = xgb.DMatrix( # Тестовая матрица
  data = as.matrix(preprocessing_data_test)
)

# Список параметров
param_list = list(
  objective = "binary:logistic",
  eta = 0.01,
  gamma = 1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.5
)

# Устанавливаем сид
set.seed(112)

# Проведение кросс-валидации с 5 фолдами
xgbcv = xgb.cv(
  params = param_list,
  data = Dtrain,
  nrounds = 1000,
  nfold = 5,
  print_every_n = 10,
  early_stopping_rounds = 30,
  maximize = F
)

# Обучение модели XGBoost
xgb_model = xgb.train(
  data = Dtrain,
  params = param_list,
  nrounds = 1000,
  print_every_n = 10
)

# Обученная модель
xgb_model

# Features importance
preprocessing_data_train['label'] <- train_targets
var_imp = xgb.importance(
  feature_names = setdiff(names(preprocessing_data_train), "label"),
  model = xgb_model
)

# Построение графика важности переменных
xgb.plot.importance(
  var_imp, 
  top_n = 35
)

# Предсказывание
predictions <- predict(xgb_model, Dtest)

# Выбор treshold'а
choose_treshold <- function(test_targets, predictions) {
  tresholds <- seq(0.1, 0.7, 0.1)
  metrics <- data.frame()
  
  for (tr in tresholds) {
    ind <- as.integer(tr / 0.1)
    predictions <- predict(xgb_model, Dtest)
    predictions <- as.integer(predictions > tr)
    metrics[ind, "treshold"] <- tr
    metrics[ind, "F1-Score"] <- F1_Score(test_targets, predictions)
    metrics[ind, "Recall"] <- Recall(test_targets, predictions)
    metrics[ind, "Precision"] <- Precision(test_targets, predictions)
    metrics[ind, "ROC-AUC"] <- AUC(test_targets, predictions)
  }

  plots <- list()
  names <- c("F1-Score", "Recall", "Precision", "ROC-AUC")
  
  for (n_plot in c(1:4)) {
    plots[[names[n_plot]]] <- local({
      name <- names[n_plot]
      ggplot(data = metrics, aes(x = metrics$treshold, 
                                 y = metrics[, n_plot + 1])) +
        geom_line() +
        geom_point() +
        geom_text(
          label = format(round(metrics[, n_plot + 1], 3), nsmall = 3),
          nudge_x = 0, 
          nudge_y = 0.003,
          check_overlap = F,
          label.padding=unit(0.55, "lines"),
          label.size=0.4,
          color="red"
        ) +
        scale_x_continuous(breaks = seq(from = 0.1, to = 0.7, by = 0.1)) +
        scale_y_continuous(breaks = seq(from = 0.95, to = 0.98, by = 0.01))
    })
  }
  figure <- ggarrange(plotlist = plots,
                      labels = names,
                      ncol = 2, nrow = 2)
  annotate_figure(
    figure,
    top = text_grob(
      "Метрики работы модели", 
      color = "red", 
      face = "bold", 
      size = 14
    )
  )
  return(metrics)
}

metrics <- choose_treshold(test_targets, predictions)
treshold <- 0.3

# Финальное предсказание на тестовой выборке
predictions <- as.integer(predictions > treshold)

# Финальный подсчет метрик
print_metrics <- function(test_targets, predictions) {
  print(paste("F1-Score:", F1_Score(test_targets, predictions)))
  print(paste("Recall:", Recall(test_targets, predictions)))
  print(paste("Precision:", Precision(test_targets, predictions)))
  print(paste("ROC-AUC:", AUC(test_targets, predictions)))
}

print_metrics(test_targets, predictions)