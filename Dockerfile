FROM python:3.11-slim-bookworm

COPY . /app
WORKDIR /app

RUN apt-get update -y && apt-get install git -y

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["chainlit", "run", "main.py", "-w"]
