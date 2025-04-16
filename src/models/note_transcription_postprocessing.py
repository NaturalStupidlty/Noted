import logging


class NoteTranscriptionPostprocessorModel:
    """
    Model responsible for postprocessing raw ASR transcriptions
    using the OpenAI API to correct spacing, punctuation, and minor transcription errors.
    """
    def __init__(self, client):
        """
        :param client: An initialized OpenAI API client with a `.responses.create` method.
        """
        self.client = client

    def postprocess(self, text: str) -> str:
        """
        Use the LLM to clean up ASR output: fix spacing, punctuation, and minor mistakes.
        :param text: Raw transcription string.
        :return: Corrected, human-readable text.
        """
        instructions = (
            "You are a helpful assistant that takes raw speech-to-text transcription "
            "and returns a corrected, human-readable text. "
            "Correct spacing, punctuation, capitalization, and fix any transcription errors. "
            "Return only the corrected text without any additional comments or explanations. "
        )
        try:
            llm_response = self.client.responses.create(
                model="gpt-4o-mini",
                instructions=instructions,
                input=f"Transcription: {text}",
                max_output_tokens=1024,
                temperature=0.0,
            )
            corrected = llm_response.output_text.strip()
            return corrected
        except Exception:
            logging.exception("NoteTranscriptionPostprocessorModel failed to postprocess text")
            # Fallback to original if postprocessing fails
            return text
