import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        }, timeout=30)
        r.raise_for_status()
        
        response_data = r.json()
        if "embeddings" not in response_data or not response_data["embeddings"]:
            raise ValueError("No embeddings returned from API")
        
        return response_data["embeddings"]
    except requests.exceptions.RequestException as e:
        print(f"Error creating embedding: {e}")
        raise
    except (KeyError, ValueError) as e:
        print(f"Error parsing embedding response: {e}")
        raise
def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

try:
    df = joblib.load("embeddings.joblib")
    if df.empty:
        raise ValueError("Loaded DataFrame is empty")
    if 'embedding' not in df.columns:
        raise ValueError("DataFrame missing 'embedding' column")
except FileNotFoundError:
    print("Error: embeddings.joblib file not found")
    exit(1)
except Exception as e:
    print(f"Error loading embeddings: {e}")
    exit(1)

def inference(prompt):
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }, timeout=120)
        r.raise_for_status()
        
        response = r.json()
        if "response" not in response:
            raise ValueError("No 'response' field in API response")
        
        print(response)
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error during inference: {e}")
        raise
    except (KeyError, ValueError) as e:
        print(f"Error parsing inference response: {e}")
        raise
incoming_query = input("Ask a question: ").strip()
if not incoming_query:
    print("Error: Question cannot be empty")
    exit(1)

try:
    question_embedding = create_embedding([incoming_query])[0]
except (IndexError, Exception) as e:
    print(f"Error creating question embedding: {e}")
    exit(1)

try:
    embedding_matrix = np.vstack(df['embedding'].to_list())
    question_vector = np.asarray(question_embedding, dtype=float).reshape(1, -1)

    if embedding_matrix.size == 0:
        raise ValueError("Embedding matrix is empty")

    similarities = cosine_similarity(embedding_matrix, question_vector).flatten()
    top_result = 5
    max_index = similarities.argsort()[::-1][0:top_result]
    print("Thinking...")
    new_df = df.iloc[max_index].copy()
    # Convert timestamps
    new_df["start"] = new_df["start"].apply(format_time)
    new_df["end"] = new_df["end"].apply(format_time)
except Exception as e:
    print(f"Error computing similarities: {e}")
    exit(1)



prompt = f''' You are an assistant that answers using only the provided video chunks.
    {new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
    "{incoming_query}"

Task:
- Identify relevant videos and timestamps.
- Explain whether the topic is briefly mentioned or well explained.
- Guide the user where to watch.

Rules:
- Do NOT make up information.
- Use only given timestamps.
- Be clear and direct.
- Keep answer concise.

If no relevant information is found:
→ "The playlist does not contain enough information about this topic.'''

try:
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
except IOError as e:
    print(f"Error writing prompt file: {e}")

try:
    response = inference(prompt)["response"]
    print(response)
    
    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(response)
except Exception as e:
    print(f"Error during inference or writing response: {e}")
    exit(1)
