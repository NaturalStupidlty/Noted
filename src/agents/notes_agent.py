import json
import logging
from datetime import datetime
from fastapi import HTTPException

from src.models.embeddings import generate_embedding
from src.database.schemas import NoteOut
from src.config import settings


class NoteAgent:
    """
    Agent responsible for processing user input and executing actions
    such as creating, updating, searching, or deleting notes.
    """

    # Updated instructions to include the 'delete_note' action.
    INSTRUCTIONS = (
        "You are a helpful note agent. Given a user's input, decide if it is a plain note or a request command. "
        "A plain note means the user wants to simply store a note, and you should output action 'plain_note'. "
        "If the input is a command, decide among the following actions and extract additional information if needed: "
        " - 'create_note' if the user wants to create a note from a command; "
        " - 'update_note' if the user wants to modify an existing note (search query must be used to find the relevant note id); "
        " - 'search_note' if the user wants to search for notes; "
        " - 'delete_note' if the user wants to delete a note (search query must be used to find the relevant note id). "
        "Return a valid JSON object with the following keys: 'action', 'note_text', 'note_id', and 'search_query'. "
        "If some information is not applicable, set its value to null. Do not include any additional commentary."
    )

    def __init__(self, client, db, classification_agent, search_agent):
        self.client = client
        self.db = db
        self.classification_agent = classification_agent
        self.search_agent = search_agent

    def process_input(self, text: str) -> dict:
        """
        Uses an LLM to interpret the input text and determine the required action.

        Returns:
            A dictionary with keys: 'action', 'note_text', 'note_id', and 'search_query'.
        """
        try:
            llm_response = self.client.responses.create(
                model="gpt-4o-mini",
                instructions=self.INSTRUCTIONS,
                input=f"User input: {text}",
                max_output_tokens=150,
                temperature=0.0,
            )
            response_text = llm_response.output_text.strip()
            return json.loads(response_text)
        except Exception as e:
            logging.exception("NoteAgent failed to process input, defaulting to plain note.")
            return {
                "action": "plain_note",
                "note_text": text,
                "note_id": None,
                "search_query": None
            }

    def _get_embedding(self, note_text: str) -> list:
        """
        Generates an embedding vector for the given note text.

        Args:
            note_text (str): The note text to be embedded.

        Returns:
            list: The embedding vector.

        Raises:
            HTTPException: If embedding generation fails.
        """
        try:
            embedding_response = self.client.embeddings.create(
                input=note_text,
                model="text-embedding-ada-002"
            )
            return embedding_response.data[0].embedding
        except Exception as e:
            logging.exception("Embedding generation failed")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    def _create_note(self, note_text: str) -> dict:
        note_type, topic = self.classification_agent.classify(note_text)
        created_at = datetime.utcnow().isoformat()
        new_id = self.db.count_notes() + 1
        embedding_vector = generate_embedding(self.client, note_text)

        doc = {
            "id": new_id,
            "text": note_text,
            "created_at": created_at,
            "note_type": note_type,
            "topic": topic,
            "embedding": embedding_vector,
        }
        self.db.index_note(doc, new_id)
        return {"status": "created", "note": doc}

    def _update_note(self, note_text: str, note_id: any) -> dict:
        existing_note = self.db.get_note_by_id(note_id)
        if not existing_note:
            raise HTTPException(status_code=404, detail="Note not found.")

        note_type, topic = self.classification_agent.classify(note_text)
        embedding_vector = generate_embedding(self.client, note_text)
        update_fields = {
            "text": note_text,
            "note_type": note_type,
            "topic": topic,
            "embedding": embedding_vector,
        }
        updated_doc = self.db.update_note_document(note_id, update_fields)
        return {"status": "updated", "note": updated_doc}

    def _locate_note_id(self, note_text: str, note_id: any, search_query: any) -> any:
        """
        Determines the note ID.

        Args:
            note_text (str): The note text.
            note_id (any): The note ID provided (if any).
            search_query (any): The search query to locate the note.

        Returns:
            The located note ID.

        """
        if note_id is not None:
            return note_id

        search_query = search_query or note_text
        results = self.search_agent.search_notes(search_query, n=1)
        if results:
            return results[0].id
        logging.warning("No note found to delete.")
        return None

    def _delete_note(self, note_text: str, note_id: any, search_query: any) -> dict:
        """
        Deletes an existing note from the database after locating it.

        Args:
            note_text (str): The note text (used if a search is needed).
            note_id (any): The note ID provided (if any).
            search_query (any): The search query to locate the note.

        Returns:
            dict: A dictionary with status "deleted" and the deleted note document.

        Raises:
            HTTPException: If the note is not found.
        """
        note_id = self._locate_note_id(note_text, note_id, search_query)
        if note_id is None:
            logging.warning("Note to delete not found.")
            return {"status": "not_found", "note": None}

        note_to_delete = self.db.get_note_by_id(note_id)

        if not note_to_delete:
            logging.warning("Note to delete not found.")
            return {"status": "not_found", "note": None}

        self.db.delete_note_document(note_id)
        return {"status": "deleted", "note": note_to_delete}

    def _search_notes(self, query: str) -> dict:
        """
        Searches for notes matching the given query.

        Args:
            query (str): The search query string.

        Returns:
            dict: A dictionary with status "search" and a list of found notes.
        """
        filtered_notes = self.search_agent.search_notes(query, n=settings.DEFAULT_SEARCH_CANDIDATES)
        if not filtered_notes:
            logging.warning("No relevant notes found for search query.")
        note_out_list = [
            NoteOut(**item) if isinstance(item, dict) else item
            for item in filtered_notes
        ]
        return {"status": "search", "results": note_out_list}

    def handle_request(self, text: str) -> dict:
        """
        Decides what action to take based on the processed input.

        Args:
            text (str): The user-provided input.

        Returns:
            dict: A response dictionary with a status key and relevant data.
        """
        decision = self.process_input(text)
        action = decision.get("action", "plain_note")

        if action in ["plain_note", "create_note"]:
            note_text = decision.get("note_text") or text
            return self._create_note(note_text)

        elif action == "update_note":
            note_text = decision.get("note_text", "")
            provided_note_id = decision.get("note_id")
            search_query = decision.get("search_query")
            note_id = self._locate_note_id(note_text, provided_note_id, search_query)
            if note_id is None:
                return {"status": "not_found", "note": None}
            return self._update_note(note_text, note_id)

        elif action == "delete_note":
            note_text = decision.get("note_text", "")
            provided_note_id = decision.get("note_id")
            search_query = decision.get("search_query")
            return self._delete_note(note_text, provided_note_id, search_query)

        elif action == "search_note":
            search_query = decision.get("search_query") or text
            return self._search_notes(search_query)

        else:
            raise HTTPException(status_code=400, detail="Invalid action detected by NoteAgent")
