import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from elasticsearch import Elasticsearch
from openai import OpenAI

from src.config import settings
from src.database.schemas import NoteCreate, NoteOut
from src.database.notes_db import NotesDB
from src.agents.notes_agent import NoteAgent
from src.models.classification import NotesClassificationModel
from src.models.search import NotesSearchModel

# --- Initialization ---
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ES_HOST = settings.ES_HOST
INDEX_NAME = settings.INDEX_NAME
EMBEDDINGS_DIMENSION = settings.EMBEDDINGS_DIMENSION

es = Elasticsearch(ES_HOST)
notes_db = NotesDB(es=es, index_name=INDEX_NAME, embeddings_dimension=EMBEDDINGS_DIMENSION)

# Instantiate existing agents.
classification_agent = NotesClassificationModel(client=client, db=notes_db)
search_agent = NotesSearchModel(client=client, db=notes_db)

# Instantiate our NoteAgent.
note_agent = NoteAgent(
	client=client,
	db=notes_db,
	classification_agent=classification_agent,
	search_agent=search_agent,
)

# --- FastAPI Application Setup ---
app = FastAPI(title="Noted")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=FileResponse)
def get_frontend():
	return FileResponse("static/index.html")


@app.get("/topics", response_model=List[str])
def get_topics():
	"""
	Retrieve all unique topics.
	"""
	topics = notes_db.get_unique_topics()
	return topics


@app.post("/notes/create")
def handle_note_agent(note: NoteCreate):
	"""
	Processes the note request using the NoteAgent and returns a proper response
	depending on the action detected. Handles all corner cases, including if no matching
	note is found for search, update, or delete operations.
	"""
	result = note_agent.handle_request(note.text)
	status = result.get("status")

	if status in ["created", "updated"]:
		# When a new note is created or an existing note is updated,
		# return the note data.
		es.indices.refresh(index=INDEX_NAME)
		return NoteOut(**result["note"])

	elif status == "deleted":
		# When a note is deleted, ensure there is an actual note reference.
		if not result.get("note"):
			raise HTTPException(status_code=404, detail="Note not found for deletion.")
		es.indices.refresh(index=INDEX_NAME)
		return NoteOut(**result["note"])

	elif status == "not_found":
		# For update or delete operations where the note could not be located.
		raise HTTPException(status_code=404, detail="Note not found for update or deletion.")

	elif status == "search":
		# For search operations, validate that some notes were found.
		search_results = result.get("results", [])
		if not search_results:
			raise HTTPException(status_code=404, detail="No notes found for the search query.")
		return search_results

	else:
		# If the action is unrecognized, return a 400 Bad Request error.
		raise HTTPException(status_code=400, detail="Unrecognized action from NoteAgent")


@app.put("/notes/{note_id}", response_model=NoteOut)
def update_note(note_id: int, note: NoteCreate):
	"""
	Update an existing note with new text, reclassifying and recomputing its embedding.
	"""
	existing_note = notes_db.get_note_by_id(note_id)
	if not existing_note:
		raise HTTPException(status_code=404, detail="Note not found.")

	note_type, topic = classification_agent.classify(note.text)
	logging.info("Updated note classified as '%s' with topic '%s'", note_type, topic)

	try:
		embedding_response = client.embeddings.create(
			input=note.text,
			model="text-embedding-ada-002"
		)
		embedding_vector = embedding_response.data[0].embedding
	except Exception as e:
		logging.exception("Embedding generation failed")
		raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

	update_fields = {
		"text": note.text,
		"note_type": note_type,
		"topic": topic,
		"embedding": embedding_vector,
	}
	updated_doc = notes_db.update_note_document(note_id, update_fields)
	return NoteOut(**updated_doc)


@app.delete("/notes/{note_id}")
def delete_note(note_id: int):
	"""
	Delete a note by its ID.
	"""
	note = notes_db.get_note_by_id(note_id)
	if not note:
		raise HTTPException(status_code=404, detail="Note not found.")
	notes_db.delete_note_document(note_id)
	return {"message": "Note deleted successfully."}


@app.get("/notes/recent", response_model=List[NoteOut])
def get_recent_notes(n: int = Query(5, description="Number of recent notes to return")):
	"""
	Retrieve the most recent n notes.
	"""
	recent_notes = notes_db.get_recent_notes(n)
	return [NoteOut(**note) for note in recent_notes]


@app.get("/notes/all", response_model=List[NoteOut])
def get_all_notes(topic: Optional[str] = None):
	"""
	Retrieve all notes, optionally filtering by topic.
	"""
	all_notes = notes_db.get_all_notes(topic)
	return [NoteOut(**note) for note in all_notes]


@app.get("/notes/search", response_model=List[NoteOut])
def search_notes(query: str, n: int = Query(settings.DEFAULT_SEARCH_CANDIDATES,
											description="Number of candidate notes to retrieve")):
	"""
	Search for notes by delegating to the SearchAgent.
	"""
	filtered_notes = search_agent.search_notes(query, n)
	if not filtered_notes:
		logging.warning("No relevant notes found after agent filtering.")
	return filtered_notes
