# Проект: Бинарная классификация для предиктивного обслуживания оборудования

## Описание проекта
Цель проекта — разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (Target = 1) или нет (Target = 0). Результаты работы оформлены в виде Streamlit-приложени

## Датасет
Используется датасет **"AI4I 2020 Predictive Maintenance Dataset"**, содержащий 10 000 записей с 14 признаками. Подробное описание датасета можно найти в [документации](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+data).

## Установка и запуск
1. Клонируйте репозиторий:

  git clone https://github.com/IgoshinYaroslav/predictive_maintenance_project.git

2. Переключитесь на директорию:

  cd predictive_maintenance_project

3. Установите зависимости:

  pip install -r requirements.txt

4. Запустите приложение:

  streamlit run app.py

## Структура репозитория
- `app.py`: Основной файл приложения.
- `analysis_and_model.py`: Страница с анализом данных и моделью.
- `presentation.py`: Страница с презентацией проекта.
- `requirements.txt`: Файл с зависимостями.
- `data/`: Папка с данными.
- `README.md`: Описание проекта.

## Видео-демонстрация
[Ссылка на видео](video/demo.mp4)
