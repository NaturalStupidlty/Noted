import logging
from typing import Tuple


class NotesClassificationModel:
    """
    Model responsible for classifying note content via the database abstraction.
    """
    def __init__(self, client, db):
        self.client = client
        self.db = db

    def classify(self, text: str) -> Tuple[str, str]:
        # Use the database class to fetch unique topics.
        try:
            unique_topics = self.db.get_unique_topics()
        except Exception as e:
            logging.exception("ClassificationAgent failed to retrieve topics from the database")
            unique_topics = []
        topics_context = ", ".join(unique_topics) if unique_topics else "None"
        instructions = (
            "You are a helpful assistant that classifies text entries. "
            "Below is a list of current topics from previous notes: "
            f"{topics_context}. "
            "Given the note content, identify if it is a 'note' or a 'todo', and then determine "
            "an appropriate short topic. If the list is short or none of the topics match well, propose a new one. "
            "If the note contains an action that needs to be done, classify it as 'todo'. "
            "If the note is a general observation or information, classify it as 'note'. "
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
