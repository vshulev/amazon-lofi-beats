import os

from pinecone import Pinecone, ServerlessSpec 


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

pc.create_index(
    name="amazon",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)
