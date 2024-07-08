FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y 

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

CMD ["python", "app.py"]

