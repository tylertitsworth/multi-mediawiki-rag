FROM ollama/ollama

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    git \
    python3 \
    python3-pip \
    sqlite3

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    TOKENIZERS_PARALLELISM=true

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN ollama serve & \
    sleep 5 && \
    ollama create volo -f ./Modelfile

RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

ENTRYPOINT []
CMD ["bash", "-c", "/bin/ollama serve & chainlit run app.py -h -d --port 7860"]
