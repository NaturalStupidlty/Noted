import json
import logging
from datetime import datetime
from fastapi import HTTPException

from src.schemas import NoteOut
from src.config import settings


class NoteAgent:
	"""
	Agent that determines if the input is a plain note or a command (note creation, update, or search) and takes appropriate actions.
	"""

	def __init__(self, client, db, classification_agent, search_agent):
		self.client = client
		self.db = db
		self.classification_agent = classification_agent
		self.search_agent = search_agent

	def process_input(self, text: str) -> dict:
		"""
		Use the LLM to interpret the input text and determine the required action.
		Returns a JSON-like dictionary with keys:
		  - action: "plain_note", "create_note", "update_note", or "search_note"
		  - note_text: the actual note text to use (if applicable)
		  - note_id: (optional) for updates
		  - search_query: (optional) in case of a search or to identify a note for update
		"""
		instructions = (
			"You are a helpful note agent. Given a user's input, decide if it is a plain note or a request command. "
			"A plain note means the user wants to simply store a note, and you should output action 'plain_note'. "
			"If the input is a command, decide among the following actions and extract additional information if needed: "
			" - 'create_note' if the user wants to create a note from a command; "
			" - 'update_note' if the user wants to modify an existing note (search query must be used to find the relevant note id); "
			" - 'search_note' if the user wants to search for notes. "
			"Return a valid JSON object with the following keys: 'action', 'note_text', 'note_id', and 'search_query'. "
			"If some information is not applicable, set its value to null. Do not include any additional commentary."
		)
		try:
			llm_response = self.client.responses.create(
				model="gpt-4o-mini",
				instructions=instructions,
				input=f"User input: {text}",
				max_output_tokens=150,
				temperature=0.0,
			)
			response_text = llm_response.output_text.strip()
			output = json.loads(response_text)
		except Exception as e:
			logging.exception("NoteAgent failed to process input, defaulting to plain note.")
			output = {
				"action": "plain_note",
				"note_text": text,
				"note_id": None,
				"search_query": None
			}
		return output

	def handle_request(self, text: str) -> dict:
		"""
		Decide what to do based on the interpreted input.
		Returns a dictionary with a 'status' key and related data.
		"""
		decision = self.process_input(text)
		action = decision.get("action", "plain_note")

		# --- Handle note creation (or plain note) ---
		if action in ["plain_note", "create_note"]:
			note_text = decision.get("note_text") or text
			note_type, topic = self.classification_agent.classify(note_text)
			created_at = datetime.utcnow().isoformat()
			current_count = self.db.count_notes()
			new_id = current_count + 1
			try:
				embedding_response = self.client.embeddings.create(
					input=note_text,
					model="text-embedding-ada-002"
				)
				embedding_vector = embedding_response.data[0].embedding
			except Exception as e:
				logging.exception("Embedding generation failed")
				raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

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

		# --- Handle note updating ---
		elif action == "update_note":
			note_text = decision.get("note_text", "")
			note_id = decision.get("note_id")
			# If note_id was not provided, try to locate it via search.
			if note_id is None:
				search_query = decision.get("search_query") or note_text
				results = self.search_agent.search_notes(search_query, n=1)
				if results:
					note_id = results[0].id
				else:
					raise HTTPException(status_code=404, detail="No note found to update.")
			# Fetch the note before updating.
			existing_note = self.db.get_note_by_id(note_id)
			if not existing_note:
				raise HTTPException(status_code=404, detail="Note not found.")
			note_type, topic = self.classification_agent.classify(note_text)
			try:
				embedding_response = self.client.embeddings.create(
					input=note_text,
					model="text-embedding-ada-002"
				)
				embedding_vector = embedding_response.data[0].embedding
			except Exception as e:
				logging.exception("Embedding generation failed")
				raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")
			update_fields = {
				"text": note_text,
				"note_type": note_type,
				"topic": topic,
				"embedding": embedding_vector,
			}
			updated_doc = self.db.update_note_document(note_id, update_fields)
			return {"status": "updated", "note": updated_doc}

		# --- Handle note search ---
		elif action == "search_note":
			search_query = decision.get("search_query") or text
			filtered_notes = self.search_agent.search_notes(search_query, n=settings.DEFAULT_SEARCH_CANDIDATES)
			if not filtered_notes:
				logging.warning("No relevant notes found for search query.")
			# Apply the search results as if a dedicated search request had been sent:
			note_out_list = []
			for item in filtered_notes:
				if isinstance(item, dict):
					note_out_list.append(NoteOut(**item))
				else:
					note_out_list.append(item)
			return {"status": "search", "results": note_out_list}

		else:
			raise HTTPException(status_code=400, detail="Invalid action detected by NoteAgent")
