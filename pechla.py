from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load env file (isme apna HuggingFace token daalna hoga: HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# Step 1: PDF load
loader = PyPDFLoader("C:/Users/Lenovo/Downloads/Kausar_Medo_Info.pdf")
docs = loader.load()

# Step 2: Split docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# Step 3: Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Vectorstore
vector_store = Chroma.from_documents(documents, embeddings, collection_name="kausar_medicos")

# Step 5: Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Step 6: HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",   # ek strong open-source LLM
    temperature=0.3,
    max_new_tokens=512
)

# Step 7: RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 8: Query
query = "What are Kausar Medo timings?"
answer = qa_chain.run(query)

print("\n🔎 Query:", query)
print("✅ Answer:", answer)
