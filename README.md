# ML_project1 — Предсказание стоимости жилья

В этом репозитории — компактный, сквозной ноутбук для прогнозирования цен на квартиры/дома по данным объявлений. Проект показывает загрузку данных, EDA, генерацию признаков, обучение моделей и оценку качества — всё внутри одного Jupyter‑ноутбука.

## Структура репозитория
```
.
├── datasets/           # сюда поместите сырые/обработанные CSV
└── Solution.ipynb      # основной ноутбук с полной логикой
```
> Если данные большие, используйте Git LFS или добавьте их в `.gitignore`.

## Быстрый старт

### 1) Виртуальная среда
```bash
# Рекомендуется Python 3.10+
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (Powershell)
.venv\Scripts\Activate.ps1
```

### 2) Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3) Запуск ноутбука
```bash
jupyter lab  # или: jupyter notebook
```
Откройте `Solution.ipynb` и запускайте ячейки последовательно сверху вниз.

## Что внутри ноутбука
- Разведочный анализ данных (EDA): проверки, пропуски, распределения.
- Инженерия признаков: флаги из текстов/булевых полей, обработка числовых, опционально полиномиальные признаки.
- Модели регрессии:
  - Linear Regression
  - Ridge / Lasso / ElasticNet
  - (опц.) PolynomialFeatures + LinearRegression
  - (опц.) SGDRegressor для больших датасетов
- Оценка качества:
  - Разделение на train/val (или KFold).
  - Метрики: MAE / RMSE / R².
  - Сравнение с простыми бэйзлайнами (например, медиана цены).

## Рекомендуемый шаблон пайплайна (без утечек)
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

numeric = ["bathrooms", "bedrooms", "price", "..."]     # подставьте свои колонки
categorical = ["doorman", "elevator", "neighborhood", "..."]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())]), numeric),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])

model = Pipeline([
    ("prep", preprocess),
    ("reg",  Ridge(alpha=1.0))   # можно заменить на Lasso/ElasticNet и т.д.
])
model.fit(X_train, y_train)
```
Главное: подгонять энкодеры/скейлеры **только на train** и применять к val/test — это исключает утечки.

## Данные
Положите CSV‑файлы в папку `datasets/` и укажите путь к ним в ноутбуке. Крупные файлы не коммитьте в репозиторий (используйте `.gitignore` или Git LFS).

## Результаты и советы
- Следите за MAE/RMSE на валидации и сравнивайте модели честно (меняйте **один** фактор за раз).
- При новых категориях на val/test используйте `OneHotEncoder(handle_unknown="ignore")` либо попробуйте кодировщики из `category_encoders` (Target/Count и др.).

## Как развивать проект
- Добавить кросс‑валидацию (KFold / GroupKFold).
- Логировать эксперименты в `mlflow` или `wandb`.
- Экспортировать лучшую модель `joblib`‑ом и поднять простой API на FastAPI.

## Требования
См. [`requirements.txt`](requirements.txt).

## Лицензия
MIT (или другая на ваш выбор) — добавьте файл LICENSE при необходимости.
