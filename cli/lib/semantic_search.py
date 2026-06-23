from sentence_transformers import SentenceTransformer
import numpy as np, os
from lib.search_utils import MOVIE_EMBEDDINGS_PATH, load_movies

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text) -> str:
        if text == None:
            raise ValueError
        embedding = self.model.encode([text])

        return embedding[0]
    
    def build_embeddings(self, documents) -> list[float]:
        self.documents = documents
        embedding_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            embedding_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(embedding_list,show_progress_bar=True)
        np.save(MOVIE_EMBEDDINGS_PATH,self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
          self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
          if len(self.embeddings) == len(documents):
              return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if len(self.document_map) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedding = self.generate_embedding(query)
        similarity_list = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(embedding, doc_embedding)
            similarity_list.append((score, self.documents[i]))

        similarity_list.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarity_list[:limit]:
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            })
        return results

            

            

            

def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def embed_text(text) -> str:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search_instance = SemanticSearch()
    movies = load_movies()
    search_instance.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(search_instance.document_map)}")
    print(f"Embeddings shape: {search_instance.embeddings.shape[0]} vectors in {search_instance.embeddings.shape[1]} dimensions")   

def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

    
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_command(query, limit):
    search_instance = SemanticSearch()
    movies = load_movies()
    search_instance.load_or_create_embeddings(movies)
    results = search_instance.search(query, limit)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"  {result['description'][:100]}...")
        print()