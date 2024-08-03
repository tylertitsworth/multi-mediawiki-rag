#!/bin/bash

/bin/ollama serve &
sleep 5
ollama create volo -f Modelfile
chainlit run app.py -h -d --port 7860
