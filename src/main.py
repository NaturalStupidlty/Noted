import logging
import os
from datetime import datetime
from typing import List, Tuple, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from elasticsearch import Elasticsearch, NotFoundError
from openai import OpenAI

from src.config import settings
from src.schemas import NoteCreate, NoteOut

# --- Initialization ---
# OpenAI API client.
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- Elasticsearch Setup ---
ES_HOST = settings.ES_HOST
INDEX_NAME = settings.INDEX_NAME
DEFAULT_SEARCH_CANDIDATES = settings.DEFAULT_SEARCH_CANDIDATES
SEARCH_SIMILARITY_THRESHOLD = settings.SEARCH_SIMILARITY_THRESHOLD
MIN_SAMPLES_TO_USE_EMBEDDINGS_SEARCH = settings.MIN_SAMPLES_TO_USE_EMBEDDINGS_SEARCH
EMBEDDINGS_DIMENSION = settings.EMBEDDINGS_DIMENSION

es = Elasticsearch(ES_HOST)


def create_es_index():
    """
    Create the Elasticsearch index if it does not exist,
    with mapping including a dense_vector for embeddings.
    """
    if not es.indices.exists(index=INDEX_NAME):
        mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "text"},
                    "created_at": {"type": "date"},
                    "note_type": {"type": "keyword"},
                    "topic": {"type": "keyword"},
                    "embedding": {"type": "dense_vector", "dims": EMBEDDINGS_DIMENSION}
                }
            }
        }
        es.indices.create(index=INDEX_NAME, body=mapping)
        logging.info("Elasticsearch index created: %s", INDEX_NAME)
    else:
        logging.info("Elasticsearch index already exists: %s", INDEX_NAME)

create_es_index()

def get_unique_topics() -> List[str]:
    """
    Retrieve distinct topics from Elasticsearch.
    """
    try:
        agg_query = {
            "size": 0,
            "aggs": {
                "unique_topics": {
                    "terms": {"field": "topic", "size": 1000}
                }
            }
        }
        result = es.search(index=INDEX_NAME, body=agg_query)
        buckets = result.get("aggregations", {}).get("unique_topics", {}).get("buckets", [])
        return [bucket["key"] for bucket in buckets if bucket["key"]]
    except Exception:
        logging.exception("Failed to fetch unique topics from Elasticsearch")
        return []


# --- Agents ---

class ClassificationAgent:
    """
    Agent responsible for classifying note content.
    """
    def __init__(self, client, es):
        self.client = client
        self.es = es

    def classify(self, text: str) -> Tuple[str, str]:
        """
        Use the OpenAI API to classify a note as either 'note' or 'todo'
        and determine the most appropriate topic.
        """
        unique_topics = get_unique_topics()
        topics_context = ", ".join(unique_topics) if unique_topics else "None"
        instructions = (
            "You are a helpful assistant that classifies text entries. "
            "Below is a list of current topics from previous notes: "
            f"{topics_context}. "
            "Given the note content, identify if it is a 'note' or a 'todo', and then determine "
            "the most appropriate short topic from the list. "
            "If the note contains an action that needs to be done, classify it as 'todo'. "
            "If the note is a general observation or information, classify it as 'note'. "
            "If none of the topics match well, propose a new one. "
            "Return only the two answers as a comma-separated list (for example: 'todo, shopping')."
        )
        try:
            llm_response = self.client.responses.create(
                model="gpt-4o-mini",
                instructions=instructions,
                input=f"Note content: {text}",
                max_output_tokens=60,
                temperature=0.0,
            )
            result_text = llm_response.output_text.strip()
            parts = [part.strip() for part in result_text.split(",")]
            if len(parts) >= 2:
                note_type, topic = parts[0], parts[1]
            else:
                note_type, topic = "note", "uncategorized"
        except Exception:
            logging.exception("ClassificationAgent failed to classify note")
            note_type, topic = "note", "uncategorized"

        note_type = note_type.lower()
        topic = topic.lower() if topic else "uncategorized"
        if note_type not in ["note", "todo"]:
            note_type = "note"
        return note_type, topic


class SearchAgent:
    """
    Agent responsible for searching notes.
    """
    def __init__(self, client, es):
        self.client = client
        self.es = es

    def search_notes(self, query: str, n: int) -> List[NoteOut]:
        try:
            doc_count = self.es.count(index=INDEX_NAME)["count"]
        except Exception as e:
            logging.exception("SearchAgent failed to count documents in Elasticsearch")
            raise HTTPException(status_code=500, detail=f"ES count request failed: {e}")

        # Use vector search only if enough samples exist.
        if doc_count > MIN_SAMPLES_TO_USE_EMBEDDINGS_SEARCH:
            try:
                query_embedding_response = self.client.embeddings.create(
                    input=query,
                    model="text-embedding-ada-002"
                )
                query_vector = query_embedding_response.data[0].embedding
            except Exception as e:
                logging.exception("SearchAgent embedding generation failed")
                raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

            search_query = {
                "size": n,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                }
            }
            try:
                es_response = self.es.search(index=INDEX_NAME, body=search_query)
            except NotFoundError:
                logging.error("No Elasticsearch index found: %s", INDEX_NAME)
                raise HTTPException(status_code=404, detail="No notes found.")
            except Exception as e:
                logging.exception("SearchAgent vector search failed")
                raise HTTPException(status_code=500, detail=f"Elasticsearch search failed: {e}")
            hits = es_response.get("hits", {}).get("hits", [])
        else:
            # Fallback to a simple match_all search.
            search_query = {"query": {"match_all": {}}}
            try:
                es_response = self.es.search(index=INDEX_NAME, body=search_query)
            except Exception as e:
                logging.exception("SearchAgent match_all search failed")
                raise HTTPException(status_code=500, detail=f"Elasticsearch search failed: {e}")
            hits = es_response.get("hits", {}).get("hits", [])

        # Assemble candidate notes from the search results.
        candidate_notes = []
        context_lines = []
        for hit in hits:
            if doc_count > MIN_SAMPLES_TO_USE_EMBEDDINGS_SEARCH:
                score = hit.get("_score", 0)
                if score < SEARCH_SIMILARITY_THRESHOLD:
                    continue
            try:
                note_id = int(hit["_id"])
                note_text = hit["_source"]["text"]
                created_at = hit["_source"].get("created_at")
                note_type = hit["_source"].get("note_type", "note")
                topic = hit["_source"].get("topic", "uncategorized")
                candidate_notes.append(
                    NoteOut(
                        id=note_id,
                        text=note_text,
                        created_at=created_at,
                        note_type=note_type,
                        topic=topic,
                    )
                )
                context_lines.append(f"{note_id}: {note_text}")
            except Exception as exc:
                logging.debug("Error processing hit: %s, error: %s", hit, exc)
                continue

        if not candidate_notes:
            logging.warning("SearchAgent found no candidate notes for filtering.")
            return []

        # LLM filtering to choose the most relevant candidate notes.
        notes_context = "\n".join(context_lines)
        try:
            llm_response = self.client.responses.create(
                model="gpt-4o-mini",
                instructions=(
                    "You must filter candidate notes based on relevance. "
                    "Each note is provided in the format 'id: note text'. "
                    "Return only the IDs of the notes (as integers) that are highly relevant to the query. "
                    "Return the numbers as a comma-separated list (e.g., '1,3,5'), with no additional text."
                ),
                input=f"Candidate Notes:\n{notes_context}\n\nUser Query: \"{query}\"",
                max_output_tokens=100,
            )
            result_text = llm_response.output_text
        except Exception as e:
            logging.exception("SearchAgent LLM request failed")
            raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")

        try:
            filtered_ids = [int(x.strip()) for x in result_text.split(",") if x.strip().isdigit()]
        except Exception as e:
            logging.exception("Error parsing note IDs from LLM response")
            filtered_ids = []

        filtered_notes = [note for note in candidate_notes if note.id in filtered_ids]
        return filtered_notes


# --- FastAPI Application Setup ---
app = FastAPI(title="Noted")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instantiate agents.
classification_agent = ClassificationAgent(client=client, es=es)
search_agent = SearchAgent(client=client, es=es)


@app.get("/", response_class=FileResponse)
def get_frontend():
    return FileResponse("static/index.html")


@app.get("/topics", response_model=List[str])
def get_topics():
    """
    Retrieve all unique topics.
    """
    topics = get_unique_topics()
    return topics


@app.post("/notes", response_model=NoteOut)
def create_note(note: NoteCreate):
    """
    Create a new note by classifying its content and indexing it.
    """
    note_type, topic = classification_agent.classify(note.text)
    logging.info("Note classified as '%s' with topic '%s'", note_type, topic)

    created_at = datetime.utcnow().isoformat()

    # Generate a new note id.
    try:
        count_result = es.count(index=INDEX_NAME)
        current_count = count_result.get("count", 0)
    except Exception:
        logging.exception("Failed to count documents in Elasticsearch")
        current_count = 0
    new_id = current_count + 1

    # Generate the note embedding.
    try:
        embedding_response = client.embeddings.create(
            input=note.text,
            model="text-embedding-ada-002"
        )
        embedding_vector = embedding_response.data[0].embedding
    except Exception as e:
        logging.exception("Embedding generation failed")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    doc = {
        "id": new_id,
        "text": note.text,
        "created_at": created_at,
        "note_type": note_type,
        "topic": topic,
        "embedding": embedding_vector,
    }

    try:
        es.index(index=INDEX_NAME, id=new_id, body=doc, refresh="wait_for")
    except Exception as e:
        logging.exception("Failed to index note in Elasticsearch")
        raise HTTPException(status_code=500, detail=f"Elasticsearch indexing failed: {e}")

    return NoteOut(**doc)


@app.put("/notes/{note_id}", response_model=NoteOut)
def update_note(note_id: int, note: NoteCreate):
    """
    Update an existing note with new text, reclassifying and recomputing its embedding.
    """
    try:
        existing_note = es.get(index=INDEX_NAME, id=note_id)["_source"]
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Note not found.")
    except Exception as e:
        logging.exception("Error fetching note")
        raise HTTPException(status_code=500, detail=f"Error fetching note: {e}")

    # Reclassify note.
    note_type, topic = classification_agent.classify(note.text)
    logging.info("Updated note classified as '%s' with topic '%s'", note_type, topic)

    # Recompute the embedding.
    try:
        embedding_response = client.embeddings.create(
            input=note.text,
            model="text-embedding-ada-002"
        )
        embedding_vector = embedding_response.data[0].embedding
    except Exception as e:
        logging.exception("Embedding generation failed")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    update_doc = {
        "doc": {
            "text": note.text,
            "note_type": note_type,
            "topic": topic,
            "embedding": embedding_vector,
        }
    }

    try:
        es.update(index=INDEX_NAME, id=note_id, body=update_doc, refresh="wait_for")
        updated_note = es.get(index=INDEX_NAME, id=note_id)["_source"]
    except Exception as e:
        logging.exception("Failed to update note in Elasticsearch")
        raise HTTPException(status_code=500, detail=f"Elasticsearch update failed: {e}")

    return NoteOut(**updated_note)


@app.get("/notes/recent", response_model=List[NoteOut])
def get_recent_notes(n: int = Query(5, description="Number of recent notes to return")):
    """
    Retrieve the most recent n notes.
    """
    search_query = {
        "size": n,
        "query": {"match_all": {}},
        "sort": [{"created_at": {"order": "desc"}}]
    }
    try:
        response = es.search(index=INDEX_NAME, body=search_query)
    except Exception as e:
        logging.exception("Elasticsearch search for recent notes failed")
        raise HTTPException(status_code=500, detail=f"Elasticsearch search failed: {e}")

    hits = response.get("hits", {}).get("hits", [])
    notes = [NoteOut(**hit.get("_source", {})) for hit in hits]
    return notes


@app.get("/notes/all", response_model=List[NoteOut])
def get_all_notes(topic: Optional[str] = None):
    """
    Retrieve all notes, optionally filtering by topic.
    """
    if topic:
        search_query = {"query": {"term": {"topic": topic.lower()}}}
        logging.debug("Fetching notes for topic: %s", topic)
    else:
        search_query = {"query": {"match_all": {}}}

    try:
        response = es.search(index=INDEX_NAME, body=search_query, scroll='2m')
    except Exception as e:
        logging.exception("Elasticsearch search for all notes failed")
        raise HTTPException(status_code=500, detail=f"Elasticsearch search failed: {e}")

    hits = response.get("hits", {}).get("hits", [])
    notes = [NoteOut(**hit.get("_source", {})) for hit in hits]
    return notes


@app.get("/notes/search", response_model=List[NoteOut])
def search_notes(query: str, n: int = Query(DEFAULT_SEARCH_CANDIDATES, description="Number of candidate notes to retrieve")):
    """
    Search for notes by delegating to the SearchAgent.
    """
    logging.debug("Received search query: '%s'", query)
    filtered_notes = search_agent.search_notes(query, n)
    if not filtered_notes:
        logging.warning("No relevant notes found after agent filtering.")
    return filtered_notes
