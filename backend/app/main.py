from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from completion.completion import Completer


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

completer = Completer(
    provider="together",
)

@app.post("/api/chat")
def chat(chat_message: ChatMessage):
    response = completer.complete(chat_message.message)
    return {"response": response}