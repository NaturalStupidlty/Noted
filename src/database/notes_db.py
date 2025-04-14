import logging
from typing import List, Optional
from fastapi import HTTPException
from elasticsearch import Elasticsearch, NotFoundError


class NotesDB:
    """
    A class to encapsulate all database (Elasticsearch) interactions.
    """
    def __init__(self, es: Elasticsearch, index_name: str, embeddings_dimension: int):
        self.es = es
        self.index_name = index_name
        self.embeddings_dimension = embeddings_dimension
        self.create_index()

    def create_index(self):
        """
        Create the Elasticsearch index with the appropriate mapping if it does not exist.
        """
        if not self.es.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "integer"},
                        "text": {"type": "text"},
                        "created_at": {"type": "date"},
                        "note_type": {"type": "keyword"},
                        "topic": {"type": "keyword"},
                        "embedding": {"type": "dense_vector", "dims": self.embeddings_dimension}
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping)
            logging.info("Elasticsearch index created: %s", self.index_name)
        else:
            logging.info("Elasticsearch index already exists: %s", self.index_name)

    def get_unique_topics(self) -> List[str]:
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
            result = self.es.search(index=self.index_name, body=agg_query)
            buckets = result.get("aggregations", {}).get("unique_topics", {}).get("buckets", [])
            return [bucket["key"] for bucket in buckets if bucket["key"]]
        except Exception:
            logging.exception("Failed to fetch unique topics from Elasticsearch")
            return []

    def count_notes(self) -> int:
        """
        Count the number of notes in the index.
        """
        try:
            result = self.es.count(index=self.index_name)
            return result.get("count", 0)
        except Exception:
            logging.exception("Failed to count documents in Elasticsearch")
            return 0

    def index_note(self, doc: dict, note_id: int):
        """
        Index a new note document.
        """
        try:
            self.es.index(index=self.index_name, id=note_id, body=doc, refresh="wait_for")
        except Exception as e:
            logging.exception("Failed to index note in Elasticsearch")
            raise HTTPException(status_code=500, detail=f"Elasticsearch indexing failed: {e}")

    def update_note_document(self, note_id: int, update_fields: dict) -> dict:
        """
        Update an existing note with new fields.
        """
        try:
            self.es.update(index=self.index_name, id=note_id, body={"doc": update_fields}, refresh="wait_for")
            updated_note = self.es.get(index=self.index_name, id=note_id)["_source"]
            return updated_note
        except Exception as e:
            logging.exception("Failed to update note in Elasticsearch")
            raise HTTPException(status_code=500, detail=f"Elasticsearch update failed: {e}")

    def get_recent_notes(self, n: int) -> List[dict]:
        """
        Retrieve the most recent n notes.
        """
        search_query = {
            "size": n,
            "query": {"match_all": {}},
            "sort": [{"created_at": {"order": "desc"}}]
        }
        try:
            response = self.es.search(index=self.index_name, body=search_query)
            hits = response.get("hits", {}).get("hits", [])
            return [hit.get("_source", {}) for hit in hits]
        except Exception as e:
            logging.exception("Elasticsearch search for recent notes failed")
            raise HTTPException(status_code=500, detail=f"Elasticsearch search failed: {e}")

    def get_all_notes(self, topic: Optional[str] = None) -> List[dict]:
        """
        Retrieve all notes or filter by a specific topic.
        """
        if topic:
            search_query = {"query": {"term": {"topic": topic.lower()}}}
            logging.debug("Fetching notes for topic: %s", topic)
        else:
            search_query = {"query": {"match_all": {}}}

        try:
            response = self.es.search(index=self.index_name, body=search_query, scroll='2m')
            hits = response.get("hits", {}).get("hits", [])
            return [hit.get("_source", {}) for hit in hits]
        except Exception as e:
            logging.exception("Elasticsearch search for all notes failed")
            raise HTTPException(status_code=500, detail=f"Elasticsearch search failed: {e}")

    def get_note_by_id(self, note_id: int) -> Optional[dict]:
        """
        Retrieve a note by its ID.
        """
        try:
            note = self.es.get(index=self.index_name, id=note_id)["_source"]
            return note
        except NotFoundError:
            return None
        except Exception as e:
            logging.exception("Error fetching note")
            raise HTTPException(status_code=500, detail=f"Error fetching note: {e}")

    def search_notes(self, query: str, n: int) -> List[dict]:
        """
        Perform a simple full-text search for notes.
        """
        search_query = {
            "size": n,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "topic"]
                }
            }
        }
        try:
            response = self.es.search(index=self.index_name, body=search_query)
            hits = response.get("hits", {}).get("hits", [])
            return [hit.get("_source", {}) for hit in hits]
        except Exception as e:
            logging.exception("Elasticsearch search for notes failed")
            raise HTTPException(status_code=500, detail=f"Elasticsearch search failed: {e}")
