from pymilvus import MilvusClient
from utils import create_embedded_vector

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
    "Sai is fat. She is a girl and wear glasses.",
    "Phone Number of Sai is 123456"
]

def build_knowledge_base():
    client = MilvusClient("milvus_demo.db")
    
    if client.has_collection(collection_name="demo_collection"):
        client.drop_collection(collection_name="demo_collection")
    client.create_collection(
        collection_name="demo_collection",
        dimension=3072,
    )
    
    data = [
        {"id": i, "vector": create_embedded_vector(docs[i]), "text": docs[i], "subject": "history"}
        for i in range(len(docs))
    ]
    
    res = client.insert(collection_name="demo_collection", data=data)
    print(res)
    
if __name__ == "__main__":
    build_knowledge_base()