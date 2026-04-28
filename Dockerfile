FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY rental_price_mlops ./rental_price_mlops

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "rental_price_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]