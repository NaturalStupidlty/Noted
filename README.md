# **_Noted_**

## 0. Description

The NoteAgent determines if it is just a note text or a request.
If it is just a note, it is classified and is written into db. 
If it is a request, the NoteAgent determines the type of request and takes an action: it can either be note creation,
note editing, or note search. 

### AI powered search:
- 1. [Elasticsearch](https://www.elastic.co/elasticsearch
) with Ada embeddings (high quality search but retrieves lots of likely relevant samples). 
This step may be skipped if there are few records.
- 2. LLM filtering of less than N results (advanced filtering logic based on the user query)


### Classification:
The items are classified using AI into 2 categories for now:
- 1. Notes
- 2. TODOs

### AI based topic extraction:
The topic for each notes is AI generated. 
The model is fed all unique topics that are present in the notes to be aware of context and
have fewer duplicate topics.

Future improvements: Automatically extract structured data from the notes and store it in a structured format.
Use context of recent notes to improve the quality of results.

## 1. Installation

- 1. Clone the repository
- 2. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```

- 3. Install docker and run Elasticsearch container
```bash
sudo apt install docker.io
sudo usermod -aG docker $USER
newgrp docker
```

## 2. Running

```bash
uvicorn src.main:app --reload
```

```bash
# Disabling security must NOT be done in production, this is only for local development
sudo docker run -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "xpack.security.http.ssl.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.6.0
```
App running on http://127.0.0.1:8000 (Press CTRL+C to quit)
