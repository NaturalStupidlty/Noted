import logging
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from whisper_streaming.whisper_online import (OpenaiApiASR, FasterWhisperASR,
                                              OnlineASRProcessor)

router = APIRouter()


@router.websocket("/ws/voice")
async def voice_transcription(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice transcription using whisper_streaming.
    Receives raw audio chunks (16kHz, mono, S16_LE) from the client,
    converts them to a NumPy array of type int16, and sends back non-empty partial
    transcriptions. When the client disconnects, the final transcription is attempted.
    """
    await websocket.accept()
    logging.info("Voice WebSocket connected")

    # Initialize the Whisper ASR model.
    asr = OpenaiApiASR("en", 0.0)
    #asr = FasterWhisperASR("en", "large-v2")

    # Create a streaming processor.
    online = OnlineASRProcessor(asr)

    try:
        while True:
            # Receive audio chunk as raw bytes.
            audio_chunk = await websocket.receive_bytes()
            # Convert raw bytes to a NumPy array of type int16, normalize to float32 in range [-1,1].
            # Insert chunk into the streaming processor.
            online.insert_audio_chunk(np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0)

            # Process and send all available partial outputs.
            for partial in online.process_iter():
                # Convert the output to text.
                text = str(partial).strip()
                # Skip sending if text is empty or "None".
                if not text or text.lower() == "none":
                    continue

                # Filter out numeric values (which are likely timing outputs).
                try:
                    _ = float(text)
                    # If conversion to float succeeds, this is a numeric value; skip it.
                    continue
                except ValueError:
                    # Not a numeric value; send it as transcription text.
                    print(text)
                    try:
                        await websocket.send_text(text)
                        logging.info("Sent transcription: %s", text)
                    except Exception as e:
                        logging.info("Error sending partial transcription: %s", e)
                        return

    except WebSocketDisconnect:
        logging.info("Client disconnected. Finalizing transcription...")
        try:
            final_result = online.finish()
            final_text = str(final_result).strip()
            if final_text and final_text.lower() != "none":
                # Also filter out numeric values in the final result.
                try:
                    _ = float(final_text)
                except ValueError:
                    try:
                        await websocket.send_text(final_text)
                        logging.info("Sent final transcription: %s", final_text)
                    except Exception as e:
                        logging.info("Error sending final transcription: %s", e)
        except Exception as e:
            logging.exception("Error during final transcription: %s", e)
        finally:
            try:
                await websocket.close()
            except Exception as e:
                logging.info("Error closing WebSocket: %s", e)
