import speech_recognition as sr
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import pyttsx3
from dotenv import load_dotenv
import os
import json
import time
import threading
from log_setup import setup_logger

load_dotenv()
logger = setup_logger()

CHROMA_PATH = "chroma"
CACHE_FILE = "query_cache.json"

PROMPT_TEMPLATE = """
Based on the following context, provide a concise, 1-2 sentence response (max 30 words) to address the customer's question directly. Use general sales techniques if context is incomplete:

{context}

---

Customer question: {question}
"""

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def save_text_to_audio(text: str):
    try:
        engine = pyttsx3.init("sapi5")  # Use sapi5 for Windows
        engine.setProperty("rate", 150)  # Natural speaking rate
        engine.setProperty("volume", 0.9)  # Loud and clear

        # Preferred female voices for customer service
        preferred_voices = ["zira", "hazel", "eva", "susan"]

        voices = engine.getProperty("voices")
        selected = False
        for voice in voices:
            if any(name in voice.name.lower() for name in preferred_voices):
                engine.setProperty("voice", voice.id)
                logger.info(f"Selected voice: {voice.name}")
                selected = True
                break

        if not selected:
            logger.warning("Preferred female voice not found. Using default voice.")

        logger.info(f"Speaking text: {text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")

def invoke_with_retry(model, prompt, max_retries=3, delay=2):
    """
    Attempts to invoke the model with retries on failure.
    Returns the response or None if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            response = model.invoke(prompt)
            logger.info("Received AI response")
            return response
        except Exception as e:
            logger.warning(f"AI invocation failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    return None

def get_immediate_response(query_text):
    """
    Generates a quick, generic response based on keywords in the query text.
    """
    query_lower = query_text.lower()
    if "property" in query_lower and "sell" in query_lower:
        return "Great, let’s talk about selling property!"
    elif "property" in query_lower:
        return "I’d love to help with your property questions!"
    elif "cold calling" in query_lower or "call" in query_lower:
        return "Cold calling, excellent! Let’s dive into that."
    elif "marketing" in query_lower:
        return "Marketing’s a great topic! Let me share some insights."
    else:
        return "Thanks for your question! I’m gathering the details for you."

def process_query(query_text, db, model, cache):
    """
    Processes the query in a background thread, returning the response and sources.
    """
    results = []
    response_text = None

    # Check cache first
    if query_text in cache:
        response_text = cache[query_text]
        logger.info(f"Using cached response for: {query_text}")
    else:
        # Search the DB with lower threshold
        logger.info("Starting similarity search")
        results = db.similarity_search_with_relevance_scores(query_text, k=4)
        if len(results) == 0 or results[0][1] < 0.2:
            response_text = (
                "I appreciate your question! My expertise is in property and cold calling, "
                "so let’s explore how I can assist you with those topics."
            )
            logger.info("No relevant results found, using fallback response")
        else:
            context_text = "\n\n---\n\n".join([doc.page_content[:500] for doc, _ in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            logger.info("Generated prompt for AI")
            response_text = invoke_with_retry(model, prompt, max_retries=3, delay=2)
            if response_text is None:
                response_text = (
                    "I’m having trouble processing that right now. Could we discuss property or cold calling? "
                    "Please share more details or try another question."
                )
                logger.info("AI response failed after retries, using fallback response")
            else:
                # Only cache valid AI responses
                cache[query_text] = response_text
                save_cache(cache)

    return response_text, results

def main():
    # Initialize recognizer
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.5  # Stop after 1.5 seconds of silence
    cache = load_cache()

    # Get the API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        raise ValueError("GOOGLE_API_KEY is required.")

    # Prepare the DB
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="semantic_similarity",
        google_api_key=api_key
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Initialize the model
    model = GoogleGenerativeAI(model="gemini-1.5-flash")

    print("Bot is ready. Speak your question (say 'exit' to stop)...")
    save_text_to_audio("Hello! I'm ready to assist you with property and cold calling queries.")
    while True:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for noise
            print("Listening...")
            audio = recognizer.listen(source, timeout=None)
        try:
            query_text = recognizer.recognize_google(audio)
            print("You said: " + query_text)
            logger.info(f"Transcribed query: {query_text}")
            if query_text.lower() == "exit":
                save_text_to_audio("Thank you for calling. Goodbye!")
                print("Exiting...")
                break

            # Immediate response
            immediate_response = get_immediate_response(query_text)
            save_text_to_audio(immediate_response)

            # Process query in background
            response_text, results = process_query(query_text, db, model, cache)

            # Speak the detailed response
            save_text_to_audio(response_text)

            # Log and print response
            sources = [doc.metadata.get("source", "None") for doc, _ in results]
            formatted_response = f"Response: {response_text}\nSources: {sources}"
            logger.info(formatted_response)
            print(formatted_response)

        except sr.UnknownValueError:
            save_text_to_audio("I’m sorry, I couldn’t catch that. Could you please repeat your question?")
            continue
        except sr.RequestError:
            save_text_to_audio("Sorry, there’s an issue with the speech service. Please try again.")
            continue

if __name__ == "__main__":
    main()