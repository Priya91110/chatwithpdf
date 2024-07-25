

from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
import os 
from dotenv import load_dotenv

load_dotenv()
openai_api_key= os.environ.get('OPENAI_API_KEY')
pdfreader = PdfReader('documents/tutorial.pdf')
from typing_extensions import Concatenate

raw_text = ' '
for i,page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content
# print(raw_text)
text_splitter = CharacterTextSplitter(
    separator= '\n',
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len
)
texts = text_splitter.split_text(raw_text)
print(len(texts))

embeddings = OpenAIEmbeddings()

document_serach = FAISS.from_texts(texts, embeddings)
print(document_serach)
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI 

chain = load_qa_chain(OpenAI(), chain_type="stuff")
query = input("Ask any question pdf")
docs = document_serach.similarity_search(query)
data = chain.run(input_documents = docs, question=query)

print(data)





# hello 

# pdfreader



'''
langchain
openai
PyPDF2
faiss-cpu
tiktoken:- creating token
'''