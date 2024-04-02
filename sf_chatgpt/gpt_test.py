import gradio as gr
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


# 문서 로드
from langchain.document_loaders import PyPDFLoader

# PDF 파일 로드
file_path = "SPRI_AI_Brief_2023년12월호_F.pdf"
loader = PyPDFLoader(file_path=file_path)
docs = loader.load()

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_docs = loader.load_and_split(text_splitter=text_splitter)

# 임베딩 & 벡터스토어 생성
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 리트리버 생성
k = 3
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = k

faiss_vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings())
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

# 프롬프트 생성
prompt = hub.pull("rlm/rag-prompt")

# 언어모델 생성
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_generator(history):
    chat_logs = []

    for item in history:
        if item[0] is not None: # 사용자
            message =  {
              "role": "user",
              "content": item[0]
            }
            chat_logs.append(message)            
        if item[1] is not None: # 챗봇
            message =  {
              "role": "assistant",
              "content": item[1]
            }
            chat_logs.append(message)            
    
    messages=[
        {
          "role": "system",
          "content": "당신은 문서분석가입니다. 사용자의 질문에 친절하게 대답하세요."
        }
    ]
    messages.extend(chat_logs)
        
    rag_chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def user(user_message, history):
    return "", history + [[user_message, None]]

def bot(history):
    rag_chain = create_generator(history)
    history[-1][1] = ""
    while True:
        response = next(rag_chain.invoke(history))
        delta = response.choices[0].delta
        if delta.content is not None:
            history[-1][1] += delta.content
        else:
            break
        yield history

chatbot = gr.Chatbot()
msg = gr.Textbox()
clear = gr.Button("Clear")

msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
    bot, chatbot, chatbot
)
clear.click(lambda: None, None, chatbot, queue=False)

gr.Interface(
    fn=bot, 
    inputs=msg, 
    outputs=chatbot,
    title="ChatGPT 기반 챗봇",
    theme="compact",
    layout="vertical"
).launch()
