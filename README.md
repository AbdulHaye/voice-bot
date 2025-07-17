# Langchain RAG Tutorial

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`.

   - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

   ```python
    conda install onnxruntime -c conda-forge
   ```

   See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additonal help if needed.

   - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.

2. Now run this command to install dependenies in the `requirements.txt` file.

```python
pip install -r requirements.txt
```

3. Install markdown depenendies with:

```python
pip install "unstructured[md]"
```

## Create database

Create the Chroma DB.

```python
python create_database.py
```

# create virtual envirment

make sure you have python 3.11x

python -3.11 -m venv .venv

# For Windows:

.venv\Scripts\activate

# For Mac/Linux:

source .venv/bin/activate

## Query the database

Query the Chroma DB.

```python
python query_data.py "How does Alice meet the Mad Hatter?"
```

the bot will stop after the person stop speaking for 1.5 seconds and then start replying

you need GOOGLE_API_KEY from https://console.cloud.google.com/ and place it in .env
you can press cntrl + c or say "exit" to stop the bot

> You'll also need to set up an OpenAI account (and set the OpenAI key in your environment variable) for this to work.

Here is a step-by-step tutorial video: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).
