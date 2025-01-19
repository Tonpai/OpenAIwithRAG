from openai import OpenAI
from pymilvus import MilvusClient
from utils import create_embedded_vector

messages = [
    {"role": "developer", "content": "You are a helpful assistant."}
]

def main():
    openai_client = OpenAI()
    db_client = MilvusClient("milvus_demo.db")
    
    while True:
        inp = input("You: ")
        print("")
        
        messages.append({ "role": "user", "content": inp })
        res = db_client.search(
            collection_name="demo_collection",
            data=[create_embedded_vector(inp)],
            limit=1, 
            output_fields=["text", "subject"]
        )
        messages.append({"role": "developer", "content": "related content: {}".format(res[0][0]["entity"]["text"])})
        
        if inp == "bye":
            break
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        ai_msg = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": ai_msg })
        print("AI:" + ai_msg + "\n")
        
        

if __name__ == "__main__":
    main()