import logging
from typing import List
from fastapi import HTTPException
from src.schemas import NoteOut
from src.config import settings


class NotesSearchModel:
    """
    Agent responsible for searching notes via the database abstraction.
    """
    def __init__(self, client, db):
        self.client = client
        self.db = db

    def search_notes(self, query: str, n: int) -> List[NoteOut]:
        # Count documents using the database interface.
        try:
            doc_count = self.db.count_notes()
        except Exception as e:
            logging.exception("SearchAgent failed to count documents in the database")
            raise HTTPException(status_code=500, detail=f"Database count request failed: {e}")

        # Use vector search only if enough samples exist.
        if doc_count > settings.MIN_SAMPLES_TO_USE_EMBEDDINGS_SEARCH:
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
                es_response = self.db.raw_search(search_query)
            except Exception as e:
                logging.exception("SearchAgent vector search failed")
                raise HTTPException(status_code=500, detail=f"Database search failed: {e}")
            hits = es_response.get("hits", {}).get("hits", [])
        else:
            # Fallback to a simple match_all search.
            search_query = {"query": {"match_all": {}}}
            try:
                es_response = self.db.raw_search(search_query)
            except Exception as e:
                logging.exception("SearchAgent match_all search failed")
                raise HTTPException(status_code=500, detail=f"Database search failed: {e}")
            hits = es_response.get("hits", {}).get("hits", [])

        # Assemble candidate notes from the search results.
        candidate_notes = []
        context_lines = []
        for hit in hits:
            if doc_count > settings.MIN_SAMPLES_TO_USE_EMBEDDINGS_SEARCH:
                score = hit.get("_score", 0)
                if score < settings.SEARCH_SIMILARITY_THRESHOLD:
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
