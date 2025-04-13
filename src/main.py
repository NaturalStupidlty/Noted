import logging
import os
from datetime import datetime
from typing import List, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from elasticsearch import Elasticsearch, NotFoundError
from openai import OpenAI

from src.config import settings
from src.schemas import NoteCreate, NoteOut

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
	Create the Elasticsearch index with a dense_vector mapping for embeddings.
	Also defines fields for text, timestamp, and classification.
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
					"embedding": {
						"type": "dense_vector",
						"dims": EMBEDDINGS_DIMENSION
					}
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
	Fetch distinct topics from Elasticsearch using an aggregation query.
	"""
	try:
		agg_query = {
			"size": 0,
			"aggs": {
				"unique_topics": {
					"terms": {
						"field": "topic",
						"size": 1000
					}
				}
			}
		}
		result = es.search(index=INDEX_NAME, body=agg_query)
		buckets = result.get("aggregations", {}).get("unique_topics", {}).get("buckets", [])
		unique_topics = [bucket["key"] for bucket in buckets if bucket["key"]]
		return unique_topics
	except Exception as e:
		logging.exception("Failed to fetch unique topics from Elasticsearch")
		return []


def classify_and_tag_note(text: str) -> Tuple[str, str]:
	"""
	Use the OpenAI API to classify a note and assign its topic.
	Retrieves all current topics from Elasticsearch so that the model may select
	one from the existing list (or propose a new one).

	The expected output is a comma-separated string, e.g., "todo, shopping".
	"""
	unique_topics = get_unique_topics()
	topics_context = ", ".join(unique_topics) if unique_topics else "None"

	instructions = (
		"You are a helpful assistant that classifies text entries. "
		"Below is a list of current topics from previous notes: "
		f"{topics_context}. "
		"Given the note content, identify if it is a 'note' or a 'todo', and then determine "
		"the most appropriate short topic from the list. If none match well, propose a new one. "
		"Return only the two answers as a comma-separated list (for example: 'todo, shopping')."
	)

	try:
		llm_response = client.responses.create(
			model="gpt-3.5-turbo",
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
	except Exception as e:
		logging.exception("Failed to classify and tag note")
		note_type, topic = "note", "uncategorized"

	note_type = note_type.lower()
	topic = topic.lower()
	if note_type not in ["note", "todo"]:
		note_type = "note"
	if not topic:
		topic = "uncategorized"

	return note_type, topic


app = FastAPI(title="Noted")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=FileResponse)
def get_frontend():
	return FileResponse("static/index.html")


@app.post("/notes", response_model=NoteOut)
def create_note(note: NoteCreate):
	"""
	Save a note: classify, generate its embedding, and store everything in Elasticsearch.
	"""
	# Classify the note.
	note_type, topic = classify_and_tag_note(note.text)
	logging.info("Note classified as '%s' with topic '%s'", note_type, topic)

	# Prepare the timestamp.
	created_at = datetime.utcnow().isoformat()

	# Generate a new note id based on the number of documents.
	try:
		count_result = es.count(index=INDEX_NAME)
		current_count = count_result.get("count", 0)
	except Exception as e:
		logging.exception("Failed to count documents in Elasticsearch for id generation")
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

	# Create the document to be indexed.
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
	Update a note with new text. This endpoint reclassifies the note and
	recomputes its embedding before updating it in Elasticsearch.
	"""
	# Check if the note exists.
	try:
		existing_note = es.get(index=INDEX_NAME, id=note_id)["_source"]
	except NotFoundError:
		raise HTTPException(status_code=404, detail="Note not found.")
	except Exception as e:
		logging.exception("Error fetching note")
		raise HTTPException(status_code=500, detail=f"Error fetching note: {e}")

	# Reclassify the updated note.
	note_type, topic = classify_and_tag_note(note.text)
	logging.info("Updated note classified as '%s' with topic '%s'", note_type, topic)

	# Recompute the embedding for the new text.
	try:
		embedding_response = client.embeddings.create(
			input=note.text,
			model="text-embedding-ada-002"
		)
		embedding_vector = embedding_response.data[0].embedding
	except Exception as e:
		logging.exception("Embedding generation failed")
		raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

	# Build the update document.
	update_doc = {
		"doc": {
			"text": note.text,
			"note_type": note_type,
			"topic": topic,
			"embedding": embedding_vector,
			# Optionally, you could add an "updated_at" timestamp here.
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
	Retrieve the most recent n notes from Elasticsearch.
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
def get_all_notes():
	"""
	Retrieve all notes from Elasticsearch.
	"""
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
def search_notes(
		query: str,
		n: int = Query(DEFAULT_SEARCH_CANDIDATES, description="Number of candidate notes to retrieve")
):
	"""
	Search for notes using a combination of Elasticsearch vector search (if enough samples exist)
	and an LLM-based filter on candidate note IDs.
	"""
	logging.debug("Entered search_notes endpoint with query: '%s' and n: %s", query, n)

	try:
		doc_count = es.count(index=INDEX_NAME)["count"]
		logging.debug("Document count from Elasticsearch: %s", doc_count)
	except Exception as e:
		logging.exception("Failed to count documents in Elasticsearch")
		raise HTTPException(status_code=500, detail=f"ES count request failed: {e}")

	if doc_count > MIN_SAMPLES_TO_USE_EMBEDDINGS_SEARCH:
		try:
			query_embedding_response = client.embeddings.create(
				input=query,
				model="text-embedding-ada-002"
			)
			query_vector = query_embedding_response.data[0].embedding
			logging.debug("Obtained query vector: %s", query_vector)
		except Exception as e:
			logging.exception("Embedding generation for query failed")
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
		logging.debug("Vector search query: %s", search_query)
		try:
			es_response = es.search(index=INDEX_NAME, body=search_query)
			logging.debug("Elasticsearch vector search response: %s", es_response)
		except NotFoundError:
			logging.error("No Elasticsearch index found: %s", INDEX_NAME)
			raise HTTPException(status_code=404, detail="No notes found.")
		except Exception as e:
			logging.exception("Elasticsearch vector search failed")
			raise HTTPException(status_code=500, detail=f"Elasticsearch search failed: {e}")
		hits = es_response.get("hits", {}).get("hits", [])
	else:
		search_query = {"query": {"match_all": {}}}
		logging.debug("Performing LLM query search on all samples with query: %s", search_query)
		try:
			es_response = es.search(index=INDEX_NAME, body=search_query)
			logging.debug("Elasticsearch match_all search response: %s", es_response)
		except Exception as e:
			logging.exception("Elasticsearch match_all search failed")
			raise HTTPException(status_code=500, detail=f"Elasticsearch search failed: {e}")
		hits = es_response.get("hits", {}).get("hits", [])

	# Build candidate notes and context for LLM filtering.
	candidate_notes = []
	context_lines = []
	for hit in hits:
		if doc_count > MIN_SAMPLES_TO_USE_EMBEDDINGS_SEARCH:
			score = hit.get("_score", 0)
			if score < SEARCH_SIMILARITY_THRESHOLD:
				logging.debug("Hit score (%s) below threshold (%s); skipping.", score, SEARCH_SIMILARITY_THRESHOLD)
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
			logging.info("Accepted candidate note: %s: %s", note_id, note_text)
		except Exception as exc:
			logging.debug("Error processing hit: %s, error: %s", hit, exc)
			continue

	if not candidate_notes:
		logging.warning("No candidate notes found for LLM filtering.")
		return []

	notes_context = "\n".join(context_lines)
	logging.debug("Constructed notes context for LLM: %s", notes_context)

	try:
		llm_response = client.responses.create(
			model="gpt-4o-mini",
			instructions="""You must filter candidate notes based on relevance.
			Each note is provided in the format 'id: note text'.
			Return only the IDs of the notes (as integers) that are highly relevant to the query.
			Return the numbers as a comma-separated list (e.g., "1,3,5"), with no additional text.""",
			input=f"""Candidate Notes:
			{notes_context}
			
			User Query: "{query}"
			""",
			max_output_tokens=100,
		)
		result_text = llm_response.output_text
		logging.debug("LLM response text: %s", result_text)
	except Exception as e:
		logging.exception("LLM request failed")
		raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")

	try:
		filtered_ids = [int(x.strip()) for x in result_text.split(",") if x.strip().isdigit()]
		logging.debug("Extracted filtered note IDs: %s", filtered_ids)
	except Exception as e:
		logging.exception("Error parsing note IDs from LLM response")
		filtered_ids = []

	filtered_notes = [note for note in candidate_notes if note.id in filtered_ids]
	logging.debug("Final filtered notes: %s", filtered_notes)
	if not filtered_notes:
		logging.warning("No relevant notes found after LLM filtering. Filtered IDs: %s", filtered_ids)
		return []

	return filtered_notes
