from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ES_HOST: str = "http://localhost:9200"
    INDEX_NAME: str = "notes"

    # Search configuration.
    DEFAULT_SEARCH_CANDIDATES: int = 100
    SEARCH_SIMILARITY_THRESHOLD: float = 1.75
    MIN_SAMPLES_TO_USE_EMBEDDINGS_SEARCH: int = 0  # Minimum number of samples to use embeddings search.
    EMBEDDINGS_MODEL: str = "text-embedding-ada-002"  # Model for generating embeddings.
    EMBEDDINGS_DIMENSION: int = 1536  # Dimension of the "text-embedding-ada-002" output.

    class Config:
        env_file = ".env"


settings = Settings()
