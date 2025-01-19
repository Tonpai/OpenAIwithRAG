from openai import OpenAI

def create_embedded_vector(text):
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding