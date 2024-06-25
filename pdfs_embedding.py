
import os
import time

from tqdm import tqdm

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)

model_name = "moka-ai/m3e-base"
model_kwargs = {"device": "cuda"}

hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs,)

directory = '/mnt/vos-6j19uo2q/code/AliRAG/bs_challenge_financial_14b_dataset/pdf'
pdf_documents = [os.path.join(directory, filename) for filename in os.listdir(directory)]

print('*******')
print(pdf_documents)


langchain_documents = []
for document in tqdm(pdf_documents):
    try:
        loader = PyPDFLoader(document)
        data = loader.load()
        langchain_documents.extend(data)
    except Exception:
        continue

print("Num pages: ", len(langchain_documents))
print("Splitting all documents")
split_docs = text_splitter.split_documents(langchain_documents)



st = time.time()
print("Embed and create vector index")
db = FAISS.from_documents(split_docs, embedding=hf)
db.save_local('faiss_index_pdf')
et = time.time()
print(f'serial embedding cost {et - st} seconds')