FROM ollama/ollama

COPY . /app
WORKDIR /app

RUN apt-get update -y && apt-get install -y \
    git \
    python3 \
    python3-pip

RUN ollama serve & sleep 5 && ollama create volo -f ./Modelfile

RUN pip install --no-cache-dir -r requirements.txt

RUN python3 main.py

RUN echo "/bin/ollama serve" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]

EXPOSE 7860

CMD ["chainlit", "run", "main.py", "-h", "--port", "7860"]
