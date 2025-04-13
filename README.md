# Noted

0. Installation

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

```bash
# Disabling security must NOT be done in production, this is only for local development
sudo docker run -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "xpack.security.http.ssl.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.6.0
```

1. Run the application
```bash
uvicorn src.main:app --reload
```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)


2. Description
https://www.elastic.co/elasticsearch

extending your application to include more complex relationships or additional data not directly tied to search, a relational database provides a scalable and structured way to model that complexity.


AI powered search:
- 1. Elasticsearch with Ada embeddings (high quality search but retrieves lots of likely relevant samples). 
This step may be skipped if there are few records.
- 2. LLM filtering of less than N results (advanced filtering logic based on the user query)


classification:
The entries are classified using AI into 2 categories for now:
- 1. Notes
- 2. TODOs

AI based topic extraction:
The model is fed all unique topic that are present in the notes to have fewer duplicate topics.