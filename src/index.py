'''
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}





'''
from fastapi import FastAPI, Request, HTTPException

from linebot import LineBotApi, WebhookHandler

from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain



app = FastAPI()
################################################################
import openai, os
	
openai.api_key = os.getenv("OPENAI_API_KEY")
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET")) 


	
conversation = []
documents =[]
for file in os.listdir("Docs"):
    if file.endswith(".pdf"):
        pdf_path = "./Docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./Docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "./Docs/" + file
        loader = TextLoader(text_path, encoding ="utf-8") #在這裡加上encoding參數否則python會報錯 目前只有txt有這個問題
        documents.extend(loader.load())

class ChatGPT:  
    

    def __init__(self):
        
        self.messages = conversation
        self.model = os.getenv("OPENAI_MODEL", default = "gpt-3.5-turbo")



    def get_response(self, user_input):
        conversation.append({"role": "user", "content": user_input})
        

        response = openai.ChatCompletion.create(
	            model=self.model,
                messages = self.messages

                )

        conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
        
        print("AI回答內容：")        
        print(response['choices'][0]['message']['content'].strip())


        
        return response['choices'][0]['message']['content'].strip()
	

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=10)
documents = text_splitter.split_documents(documents)
vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
#create the index in retriever interface
retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k":5})


chatgpt = ChatGPT()

qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.2), retriever=retriever , verbose=False , chain_type="stuff")
chat_history= []


# Line Bot config
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/callback")
async def callback(request: Request):
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Missing Parameters")
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handling_message(event):

    
    if isinstance(event.message, TextMessage):

        
        user_message = event.message.text
        result = qa({"question": user_message+'(用繁體中文回答)',"chat_history" : chat_history})
        #reply_msg = chatgpt.get_response(user_message)
        reply_msg = result['answer']

        chat_history.append((user_message,result['answer']))
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_msg))



