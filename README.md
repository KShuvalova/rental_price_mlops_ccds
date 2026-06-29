# Rental Price MLOps

Rental Price MLOps — демонстрационный MLOps-проект для предсказания цены аренды жилья на базе FastAPI.

Проект показывает полный прикладной цикл эксплуатации ML-модели в production-like условиях: подготовка данных, обучение модели, инференс через API, логирование предсказаний, обнаружение деградации, ручное переобучение, обновление модели в сервисе, мониторинг качества и сервисных метрик через web UI, Prometheus, Grafana, Kubernetes / Minikube и Argo CD.

---

## Основные возможности

* предсказание цены аренды через FastAPI API;
* OpenAPI / Swagger-документация;
* web UI для работы с моделью;
* страница инференса с ручным вводом признаков;
* таблица последних предсказаний;
* флаги аномалий и drift-уведомления;
* страница мониторинга drift и качества модели;
* ручной запуск переобучения модели через UI/API;
* страница экспериментов с метриками и артефактами;
* трекинг экспериментов и регистрация моделей в MLflow;
* версионирование данных и артефактов через DVC;
* расчет data drift, target drift и concept drift;
* генерация drift-отчетов в JSON и HTML;
* мониторинг через Prometheus;
* Grafana dashboard для технических и ML-метрик;
* контейнерный запуск через Docker и Docker Compose;
* deployment в Kubernetes / Minikube;
* CD через Argo CD по GitOps-подходу;
* CI через GitHub Actions: lint, tests, Docker build.

---

## Цель проекта

Цель проекта — спроектировать и реализовать воспроизводимую MLOps-систему, которая закрывает полный цикл работы ML-модели:

```text
новые данные
→ инференс
→ логирование предсказаний
→ расчет drift
→ уведомления о деградации
→ ручное переобучение
→ обновление модели
→ мониторинг качества и бизнес-метрик
→ deployment в Kubernetes / Minikube
→ CD через Argo CD
```

Проект служит учебным примером production-like ML-сервиса.

---

## Датасет и модель

В проекте используется датасет `AB_NYC_2019.csv` для задачи предсказания цены аренды жилья.

Базовая модель:

```text
RandomForestRegressor
```

Основные артефакты модели:

```text
models/baseline_model.pkl
reports/baseline_metrics.json
reports/test_metrics.json
reports/test_predictions.parquet
```

Пример метрик на тестовой выборке:

```json
{
  "mae_log": 0.305,
  "rmse_log": 0.435,
  "r2_log": 0.61,
  "mae_price": 58.652,
  "rmse_price": 256.661,
  "r2_price": 0.123
}
```

---

## Архитектура

Проект построен по слоистой архитектуре:

```text
API layer
  FastAPI endpoints, OpenAPI, HTML UI routes

Service layer
  загрузка модели, инференс, переобучение

Modeling layer
  подготовка данных, обучение, оценка модели

Monitoring layer
  prediction logs, current dataset, drift reports, Prometheus metrics

Experiment layer
  MLflow tracking, параметры, метрики, model registry

Infrastructure layer
  Docker, Docker Compose, Kubernetes, Minikube, Argo CD, GitHub Actions, DVC
```

---

## Используемый стек

* Python 3.11
* FastAPI
* Uvicorn
* scikit-learn
* CatBoost
* Pandas
* NumPy
* Jinja2
* MLflow
* DVC
* Prometheus
* Grafana
* Docker
* Docker Compose
* Kubernetes
* Minikube
* Argo CD
* GitHub Actions
* Ruff
* Pytest

---

## Git flow и commits

В проекте используется GitHub flow:

```text
feature branch
→ pull request / merge request
→ CI checks
→ merge в main
→ Argo CD синхронизирует manifests из main
```

Для сообщений коммитов используется формат conventional commits:

```text
feat: add minikube and argocd deployment manifests
feat: add grafana mlops monitoring dashboard
fix: format imports and optimize docker build context
docs: add minikube argocd and grafana instructions
```

---

## Структура проекта

```text
.
├── .dvc/
├── .github/
│   └── workflows/
├── argocd/
│   └── application.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── reference/
│   └── current/
├── grafana/
│   └── provisioning/
│       ├── dashboards/
│       └── datasources/
├── k8s/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   └── service.yaml
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
```

---

## Основные страницы UI

После запуска FastAPI доступны следующие страницы:

```text
/                 — страница инференса
/predictions-ui   — таблица последних предсказаний
/monitoring-ui    — drift, качество модели, флаги деградации
/experiments-ui   — модели, метрики и экспериментальные артефакты
/docs             — OpenAPI / Swagger
```

---

## Основные API endpoints

```text
GET  /health          — healthcheck
GET  /model-info      — информация о текущей модели
POST /predict         — предсказание по JSON
GET  /predictions     — последние предсказания
POST /retrain         — ручное переобучение модели
POST /retrain-ui      — запуск переобучения из UI
GET  /metrics/latest  — последние метрики модели
GET  /metrics         — Prometheus metrics
```

---

## DVC

DVC используется для версионирования данных и артефактов модели.

Проверка remote:

```bash
dvc remote list
```

В локальной конфигурации проекта DVC remote может указывать на внешний диск:

```text
/Volumes/ARCHIVE/dvc_storage
```

Полезные команды:

```bash
dvc pull --force
dvc status
dvc repro
```

Основные DVC-этапы:

```text
prepare — подготовка train/val/test/reference данных
train   — обучение baseline model и сохранение метрик
```

---

## MLflow

MLflow используется для:

* логирования параметров модели;
* логирования метрик;
* сохранения артефактов;
* регистрации модели;
* сравнения запусков обучения.

Пример локального запуска MLflow:

```bash
mlflow server \
  --backend-store-uri sqlite:////Volumes/ARCHIVE/mlflow_artifacts/mlflow.db \
  --default-artifact-root file:///Volumes/ARCHIVE/mlflow_artifacts/artifacts \
  --host 127.0.0.1 \
  --port 5000
```

MLflow UI будет доступен по адресу:

```text
http://127.0.0.1:5000
```

---

## Drift monitoring

В проекте реализован monitoring-контур для отслеживания деградации данных и качества модели.

### Data drift

Data drift — изменение распределения входных признаков.

Система сравнивает:

```text
reference dataset — данные, сформированные на этапе обучения;
current dataset   — новые данные из prediction logs.
```

Примеры признаков, по которым может отслеживаться drift:

```text
neighbourhood_group
neighbourhood
room_type
latitude
longitude
minimum_nights
number_of_reviews
reviews_per_month
availability_365
```

### Target drift

Target drift — изменение распределения целевой переменной, то есть цены аренды.

Если `actual_price` доступен, система может сравнивать реальные новые цены с обучающим распределением. Если фактический target еще недоступен, распределение `predicted_price` может использоваться как proxy-сигнал.

### Concept drift

Concept drift — изменение зависимости между признаками и целевой переменной.

В проекте concept drift отслеживается через ухудшение качества модели, когда доступны реальные значения target. Используются метрики:

```text
MAE
RMSE
R2
```

Если ошибки растут, это сигнал, что модель может требовать переобучения.

---

## Drift reports

После расчета drift формируются отчеты:

```text
reports/drift_report.json
reports/drift_report.html
```

Назначение отчетов:

```text
JSON — машинная обработка, флаги drift, отображение в UI;
HTML — визуальный анализ для ML-инженера или аналитика.
```

Дополнительные monitoring-артефакты:

```text
logs/predictions.jsonl
data/current/current_predictions.parquet
```

---

## Prometheus

FastAPI-сервис отдает метрики через endpoint:

```text
/metrics
```

Prometheus собирает метрики с сервиса `rental-api`.

Prometheus доступен по адресу:

```text
http://127.0.0.1:9090
```

Проверка targets:

```text
http://127.0.0.1:9090/targets
```

Ожидаемые targets:

```text
prometheus    UP
rental-api    UP
```

Примеры Prometheus-запросов:

```text
up
up{job="rental-api"}
scrape_duration_seconds
scrape_samples_scraped
rental_api_requests_total
rental_api_predict_requests_total
rental_api_retrain_requests_total
rental_drifted_features_count
rental_target_drift_flag
rental_concept_drift_flag
rental_current_mae_price
rental_current_rmse_price
```

---

## Grafana dashboard

Grafana используется для визуализации технических и ML-метрик, которые собирает Prometheus.

Grafana доступна по адресу:

```text
http://127.0.0.1:3000
```

Логин и пароль по умолчанию:

```text
admin / admin
```

Dashboard находится в разделе:

```text
Dashboards → Rental MLOps → Rental Price MLOps Monitoring
```

Dashboard-файлы находятся в репозитории:

```text
grafana/provisioning/dashboards/dashboard.yml
grafana/provisioning/dashboards/rental_mlops_dashboard.json
```

Dashboard содержит панели:

* `API availability` — доступность FastAPI-сервиса;
* `Request rate` — интенсивность запросов;
* `API p95 latency` — 95-й перцентиль задержки API;
* `Predict calls total` — количество вызовов инференса;
* `Retrain calls total` — количество запусков переобучения;
* `Drifted features count` — количество признаков с drift;
* `Target drift flag` — флаг target drift;
* `Concept drift flag` — флаг concept drift;
* `Current MAE price` — текущий MAE по цене;
* `Current RMSE price` — текущий RMSE по цене;
* `Current monitoring rows` — количество строк в текущем monitoring dataset.

---

## Локальный запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/KShuvalova/rental_price_mlops_ccds.git
cd rental_price_mlops_ccds
```

### 2. Создание виртуального окружения

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Подтягивание данных и артефактов через DVC

```bash
dvc pull --force
```

### 4. Проверка тестов и линтера

```bash
pytest -q
ruff check .
```

### 5. Подготовка данных и обучение модели

```bash
dvc repro
```

Или вручную:

```bash
python -m rental_price_mlops.dataset
python -m rental_price_mlops.modeling.train
python -m rental_price_mlops.modeling.evaluate
```

### 6. Запуск FastAPI

```bash
python -m uvicorn rental_price_mlops.api.main:app --reload --host 127.0.0.1 --port 8000
```

Приложение будет доступно по адресу:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Healthcheck:

```bash
curl http://127.0.0.1:8000/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

---

## Запуск через Docker Compose

Docker Compose используется для локальной отладки API, Prometheus и Grafana.

Запуск:

```bash
docker compose up --build
```

Или в фоне:

```bash
docker compose up -d --build
```

После запуска доступны:

```text
FastAPI UI:  http://127.0.0.1:8000
Swagger:     http://127.0.0.1:8000/docs
Prometheus:  http://127.0.0.1:9090
Grafana:     http://127.0.0.1:3000
```

Проверка контейнеров:

```bash
docker compose ps
```

Остановка:

```bash
docker compose down
```

---

## Kubernetes / Minikube

Финальный production-like запуск проекта реализован через Kubernetes в локальном Minikube-кластере.

Docker Compose используется для локальной отладки, а Kubernetes / Minikube используется как целевое окружение для демонстрации deployment.

### 1. Запуск Minikube

```bash
minikube start --driver=docker --memory=4000 --cpus=2 --kubernetes-version=v1.34.1
```

Проверка:

```bash
kubectl get nodes
minikube status
```

Ожидаемый результат:

```text
minikube   Ready
```

### 2. Загрузка Docker image в Minikube

После сборки Docker image нужно загрузить его в Minikube:

```bash
minikube image load rental_price_mlops_ccds-api:latest
```

Проверка:

```bash
minikube image ls | grep rental_price
```

### 3. Mount локальных данных и модели

Так как модель, данные, отчеты и логи не хранятся напрямую в Git, для локальной демонстрации они примонтируются в Minikube:

```bash
minikube mount ~/Desktop/rental_price_mlops_ccds:/mnt/rental_price_mlops_ccds
```

Этот терминал должен оставаться открытым, пока приложение работает в Minikube.

### 4. Применение Kubernetes manifests

Kubernetes manifests находятся в папке:

```text
k8s/
```

Основные файлы:

```text
k8s/namespace.yaml
k8s/deployment.yaml
k8s/service.yaml
```

Применение:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Проверка ресурсов:

```bash
kubectl get all -n rental-mlops
```

Ожидаемый результат:

```text
pod/rental-api-...      Running
service/rental-api      ClusterIP
deployment/rental-api   1/1
```

### 5. Доступ к API в Kubernetes

Для локальной проверки используется port-forward:

```bash
kubectl port-forward -n rental-mlops svc/rental-api 8000:8000
```

После этого приложение доступно по адресам:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

Проверка healthcheck:

```bash
curl http://127.0.0.1:8000/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

---

## Argo CD / GitOps CD

CD реализован через Argo CD в Minikube.

Argo CD отслеживает GitHub-репозиторий и синхронизирует Kubernetes manifests из папки `k8s/` с текущим состоянием Kubernetes-кластера.

### 1. Установка Argo CD

```bash
kubectl create namespace argocd
kubectl create -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

Проверка Pod-ов Argo CD:

```bash
kubectl get pods -n argocd
```

Ожидаемый результат:

```text
argocd-application-controller   Running
argocd-repo-server              Running
argocd-server                   Running
argocd-redis                    Running
argocd-dex-server               Running
```

### 2. Argo CD Application

Manifest Argo CD Application находится здесь:

```text
argocd/application.yaml
```

Он указывает на:

```text
repoURL: https://github.com/KShuvalova/rental_price_mlops_ccds.git
targetRevision: main
path: k8s
```

Применение:

```bash
kubectl apply -f argocd/application.yaml
```

Проверка:

```bash
kubectl get applications -n argocd
kubectl describe application rental-price-mlops -n argocd
```

Ожидаемый статус:

```text
Sync Status: Synced
Health Status: Healthy
Message: successfully synced
```

### 3. Argo CD UI

Открыть Argo CD UI:

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

Затем открыть:

```text
https://127.0.0.1:8080
```

Для локального Minikube браузер может показать предупреждение о самоподписанном сертификате. Это нормальное поведение.

Логин:

```text
admin
```

Получить пароль:

```bash
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d && echo
```

В UI приложение `rental-price-mlops` должно иметь статусы:

```text
Healthy
Synced
Sync OK
```

---

## CI/CD

В проекте настроен CI через GitHub Actions.

CI выполняет:

* lint через Ruff;
* тесты через Pytest;
* проверку `docker compose config`;
* сборку Docker image.

CD реализован через Argo CD по GitOps-подходу:

```text
изменения попадают в main
→ Argo CD отслеживает GitHub repo
→ Argo CD читает manifests из k8s/
→ Argo CD синхронизирует Minikube cluster
→ приложение получает статус Synced / Healthy
```

Таким образом, GitHub является источником желаемого состояния, а Argo CD выполняет continuous delivery в Kubernetes / Minikube.

---

## Ручное переобучение

Ручное переобучение доступно через API:

```bash
curl -X POST http://127.0.0.1:8000/retrain
```

Также переобучение можно запустить через web UI на странице мониторинга.

После переобучения обновляются:

```text
models/baseline_model.pkl
reports/baseline_metrics.json
MLflow run
```

---

## Полезные команды для защиты

Проверка Docker Compose:

```bash
docker compose ps
```

Проверка FastAPI:

```bash
curl http://127.0.0.1:8000/health
```

Проверка Prometheus targets:

```text
http://127.0.0.1:9090/targets
```

Проверка Kubernetes:

```bash
kubectl get nodes
kubectl get all -n rental-mlops
```

Проверка Argo CD:

```bash
kubectl get applications -n argocd
kubectl describe application rental-price-mlops -n argocd
```

Открыть Argo CD UI:

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

Открыть Grafana:

```text
http://127.0.0.1:3000
```

---

## Что показывать на защите

Рекомендуемый порядок демонстрации:

```text
1. FastAPI UI:
   http://127.0.0.1:8000

2. Swagger / OpenAPI:
   http://127.0.0.1:8000/docs

3. Таблица последних предсказаний:
   http://127.0.0.1:8000/predictions-ui

4. Monitoring UI:
   http://127.0.0.1:8000/monitoring-ui

5. Drift report:
   reports/drift_report.html
   reports/drift_report.json

6. MLflow:
   http://127.0.0.1:5000

7. Prometheus targets:
   http://127.0.0.1:9090/targets

8. Grafana dashboard:
   http://127.0.0.1:3000

9. Kubernetes:
   kubectl get all -n rental-mlops

10. Argo CD:
   https://127.0.0.1:8080
```

---

## Автор

Kseniya Shuvalova
