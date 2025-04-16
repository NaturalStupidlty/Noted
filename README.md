# **_Noted_**

FastAPI + OpenAI + Elasticsearch + Whisper

## 0. Description

This is a simple AI powered note taking web application.

You can text your notes or use speech recognition.

The NoteAgent determines if it is just a note text or a request.
If it is just a note, it is classified and is written into db. 
If it is a request, the NoteAgent determines the type of request and takes an action: it can either be note creation,
note editing, note deletion, or note search. 

### AI powered search:
- 1. [Elasticsearch](https://www.elastic.co/elasticsearch) 
with Ada embeddings (high quality search but retrieves lots of likely relevant samples). 
This step may be skipped if there are few records.
- 2. LLM filtering of less than N results (advanced filtering logic based on the user query)

Both manual search and AI search (just ask **Noted** to search) are available.

### AI based topic extraction:
The topic for each notes is AI generated. 
The model is fed all unique topics that are present in the notes to be aware of context and
have fewer duplicate topics.

## Filtering:

You can filter notes by:
- 1. TODO vs Note
- 2. Topic

Filtering is avalable as a dropdown menu in the web interface. 
You can also just ask **Noted** to filter the notes for you.


### Classification:
The items are classified using AI into 2 categories for now:
- 1. Notes
- 2. TODOs

### Whisper Speech Recognition
Implemented using open source [whisper_streaming](https://github.com/ufal/whisper_streaming)
Support for multiple languages, lot's of model versions, 
[quantization](https://opennmt.net/CTranslate2/quantization.html),
and streaming for real time transcription.

After the transcription is done, the text is postprocessed using AI to get more accurate results.

_____
Future improvements: Automatically extract structured data from the notes and store it in a structured format.
Use context of recent notes to improve the quality of results. Added configurable support for more languages.

## 1. Installation

- 1. Clone the repository
- 2. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```

- 3. Install docker
```bash
sudo apt install docker.io
sudo usermod -aG docker $USER
newgrp docker
```

## 2. Running

Tested on server with Ubuntu 22.04 (RTX 4090, Intel i9-13900K, 64GB RAM)

Ensure you have the following environment variables set:
```bash
export OPENAI_API_KEY=your_openai_api_key
```

Run Elasticsearch container
```bash
# Disabling security must NOT be done in production, this is only for local development
sudo docker run -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "xpack.security.http.ssl.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.6.0
```

```bash
uvicorn src.main:app --host 192.168.0.124 --port 8000 --ws websockets --reload
```

App running on http://192.168.0.124:8000 (Press CTRL+C to quit)

You can connect to the app using the web interface or using the API on any device in the same network.
