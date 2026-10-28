"""
1. Ensure docker is installed and running (https://docs.docker.com/get-docker/)
2. pip install -qU langchain_postgres
3. Run the following command to start the postgres container:
   
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
4. Use the connection string below for the postgres container

"""

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import uuid


# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# Load the document, split it into chunks
raw_documents = TextLoader('./test.txt', encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Create embeddings for the documents
embeddings_model = OpenAIEmbeddings()

db = PGVector.from_documents(
    documents, embeddings_model, connection=connection)

results = db.similarity_search("query", k=4)

print(results)

print("Adding documents to the vector store")
ids = [str(uuid.uuid4()), str(uuid.uuid4())]
db.add_documents(
    [
        Document(
            page_content="there are cats in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ],
    ids=ids,
)

print("Documents added successfully.\n Fetched documents count:",
      len(db.get_by_ids(ids)))

print("Deleting document with id", ids[1])
db.delete({"ids": ids})

print("Document deleted successfully.\n Fetched documents count:",
      len(db.get_by_ids(ids)))

"""
C:\Users\tomasz.ziss\Documents\learning-langchain\ch2\py\i-pg-vector.py:17: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
  from langchain_community.document_loaders import TextLoader
[Document(id='c2935621-0565-4997-ac2e-9b4d04fb0fb6', metadata={'source': './test.txt'}, page_content='V.'), Document(id='c4bbdf92-03bb-4911-9293-f40ac38f00f6', metadata={'source': './test.txt'}, page_content='V.'), Document(id='b79ab403-4e51-4add-a8c0-17f2e30aaf44', metadata={'source': './test.txt'}, page_content='II.'), Document(id='004e221e-48dc-4451-8b60-ef35a25bfd3f', metadata={'source': './test.txt'}, page_content='II.')]
Adding documents to the vector store
Documents added successfully.
 Fetched documents count: 2
Deleting document with id 3465e75d-20f4-446d-9a53-56db168970fd
Document deleted successfully.
 Fetched documents count: 2
"""

