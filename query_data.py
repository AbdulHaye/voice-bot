import argparse

from langchain_chroma import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI

from langchain.prompts import ChatPromptTemplate

import pyttsx3

from dotenv import load_dotenv

load_dotenv()

from log_setup import setup_logger

logger = setup_logger()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def save_text_to_audio(file_name: str, text: str):
    engine = pyttsx3.init("espeak")
    # engine.setProperty("voice", "english-us")
    voices = engine.getProperty("voices")
    volume = engine.getProperty("volume")
    rate = engine.getProperty("rate")
    logger.info(f"Voices: {voices}, Volume: {volume}, rate: {rate}")
    engine.runAndWait()


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", task_type="semantic_similarity"
    )

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = GoogleGenerativeAI(model="gemini-2.0-flash")
    response_text = model.invoke(prompt)

    # save_text_to_audio(response_text)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
