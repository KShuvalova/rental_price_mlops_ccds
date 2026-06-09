````md
# Rental Price MLOps

Rental Price MLOps — демонстрационный MLOps-проект для предсказания цены аренды жилья на базе FastAPI.

Проект показывает полный прикладной цикл работы с ML-сервисом: подготовку данных, обучение моделей, инференс через API, web UI, мониторинг drift, трекинг экспериментов, контейнерный запуск и CI.

---

## Основные возможности

- предсказание цены аренды через FastAPI API
- web UI для работы с моделью
- страница инференса с ручным вводом признаков
- страница последних предсказаний
- страница мониторинга drift и качества модели
- ручной запуск переобучения модели через UI
- страница экспериментов с метриками и артефактами
- трекинг и регистрация моделей в MLflow
- мониторинг через Prometheus и Grafana
- версионирование данных и артефактов через DVC
- CI через GitHub Actions
- контейнерный запуск через Docker и docker-compose

---

## Архитектура

Проект построен по слоистой архитектуре:

- API слой — обработка HTTP-запросов и UI routes
- сервисный слой — логика инференса, загрузки модели и переобучения
- monitoring слой — drift-отчеты, prediction logs, Prometheus metrics
- experiment layer — MLflow, метрики baseline и test, model registry
- infrastructure layer — Docker, docker-compose, GitHub Actions, DVC

Такой подход упрощает поддержку проекта и позволяет развивать его как production-like систему.

---

## Используемый стек

- Python 3.11
- FastAPI
- Uvicorn
- scikit-learn
- CatBoost
- Pandas
- NumPy
- Jinja2
- MLflow
- DVC
- Prometheus
- Grafana
- Docker
- docker-compose
- GitHub Actions
- Ruff
- Pytest

---

## Развертывание

Проект можно запускать:

- локально через uvicorn
- в Docker-контейнерах через docker-compose

Дополнительно доступны:

- MLflow для трекинга экспериментов
- Prometheus для сбора метрик
- Grafana для визуализации мониторинга

---

## Цель проекта

- показать полный прикладной MLOps-пайплайн для модели предсказания цены аренды
- продемонстрировать практику построения FastAPI-сервиса для ML-модели
- показать трекинг экспериментов и регистрацию моделей в MLflow
- показать drift-monitoring и observability через Prometheus и Grafana
- показать практики CI и контейнерного запуска
- служить учебным примером production-like ML-сервиса

---

## Структура проекта

```text
.
├── .dvc/
├── .github/workflows/
├── data/
├── grafana/
├── models/
├── notebooks/
├── prometheus/
├── rental_price_mlops/
│   ├── api/
│   ├── modeling/
│   ├── monitoring/
│   └── ui/
├── reports/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
````

---

## Основные страницы UI

После запуска web UI доступны следующие страницы:

* `/` — страница инференса
* `/predictions-ui` — последние предсказания
* `/monitoring-ui` — drift и метрики качества модели
* `/experiments-ui` — модели, метрики и экспериментальные артефакты

---

## Основные API endpoints

* `GET /health` — healthcheck
* `GET /model-info` — информация о текущей модели
* `POST /predict` — предсказание по JSON
* `GET /predictions` — последние предсказания
* `POST /retrain` — переобучение модели
* `GET /metrics/latest` — последние метрики модели
* `GET /metrics` — Prometheus metrics

---

## MLflow

В проекте реализован MLflow для:

* логирования параметров моделей
* логирования метрик
* регистрации моделей
* хранения версий моделей

Пример запуска MLflow локально:

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db
```

После запуска MLflow будет доступен по адресу:

```text
http://127.0.0.1:5000
```

---

## Мониторинг

В проекте реализован базовый monitoring-контур:

* prediction logs
* current dataset
* drift report
* Prometheus metrics
* Grafana dashboard

Основные monitoring-артефакты:

* `reports/drift_report.json`
* `reports/drift_report.html`
* `logs/`
* `/metrics` endpoint

---

## DVC

DVC используется для версионирования данных и артефактов модели.

Полезные команды:

```bash
dvc pull --force
dvc status
```

---

## Запуск проекта

### Локальный запуск

1. Клонируйте репозиторий:

```bash
git clone https://github.com/KShuvalova/rental_price_mlops_ccds.git
cd rental_price_mlops_ccds
```

2. Создайте и активируйте виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Подтяните артефакты DVC:

```bash
dvc pull --force
```

4. Запустите приложение:

```bash
uvicorn rental_price_mlops.api.main:app --reload --port 8001
```

Приложение будет доступно по адресу:

```text
http://127.0.0.1:8001
```

Swagger-документация:

```text
http://127.0.0.1:8001/docs
```

---

### Локальный запуск через Docker

Соберите и поднимите контейнеры:

```bash
docker-compose build
docker-compose up
```

После запуска будут доступны:

* FastAPI UI: `http://127.0.0.1:8000`
* Swagger: `http://127.0.0.1:8000/docs`
* Prometheus: `http://127.0.0.1:9090`
* Grafana: `http://127.0.0.1:3000`

---

## CI

В проекте настроен CI через GitHub Actions.

Проверки включают:

* lint
* tests
* build

Это позволяет автоматически валидировать изменения перед merge.

---

## Что уже реализовано

### ML и инференс

* baseline-модель Random Forest
* альтернативная модель CatBoost
* API для предсказания цены аренды
* сохранение логов предсказаний

### Мониторинг

* reference dataset и current dataset
* расчет data drift
* расчет target drift
* concept drift proxy
* генерация drift report в JSON и HTML
* экспорт метрик в Prometheus
* dashboard в Grafana

### UI

* Inference
* Predictions
* Monitoring
* Retrain
* Experiments

---

## Планы развития

* доработка README и эксплуатационных инструкций
* развертывание в Kubernetes / Minikube
* GitOps/CD через Argo CD
* расширение UI и уведомлений о drift
* более детальная визуализация экспериментов

---

## Автор

Kseniya Shuvalova

```
```
