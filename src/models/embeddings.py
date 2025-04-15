import logging
from fastapi import HTTPException

from src.config import settings


def generate_embedding(client, text: str, model: str = settings.EMBEDDINGS_MODEL) -> list:
	"""
	Generate an embedding vector using the specified OpenAI model.

	Args:
		client: The OpenAI client instance.
		text (str): The text to embed.
		model (str): The model to use for generating embeddings.

	Returns:
		list: The embedding vector.

	Raises:
		HTTPException: If the embedding generation fails.
	"""
	try:
		embedding_response = client.embeddings.create(input=text, model=model)
		return embedding_response.data[0].embedding
	except Exception as e:
		logging.exception("Embedding generation failed")
		raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")
