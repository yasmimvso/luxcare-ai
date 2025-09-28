import firebase_admin
from firebase_admin import credentials, firestore
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agents.agent import ChatbotAgent
import uuid
from datetime import datetime

# Inicializar Firebase
cred = credentials.Certificate("config/luxcare-ai-firebase.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

app = FastAPI(title="LuxCare AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ChatbotAgent()

# Modelo Pydantic para a mensagem
class Message(BaseModel):
    text: str
    user_id: str

@app.get("/")
def root():
    return {"message": "LuxCare AI API está rodando!"}

@app.post("/chat")
def chat(message: Message):

    try:
        state = {
            "messages": [{"role": "user", "content": message.text}],
            "user_id": message.user_id
        }

        result = agent.graph.invoke(state)
        conv_id = message.user_id

        assistant_messages = [m for m in result["messages"] if m.role == "assistant"]
        if not assistant_messages:
            raise HTTPException(status_code=500, detail="No assistant response")
            
        reply = assistant_messages[-1]
        response_text = reply.content

        
        user_msg_id = str(uuid.uuid4())

        db.collection("conversations").document(conv_id).collection("messages").document(user_msg_id).set({
            "sender": "user",
            "text": message.text,
            "timestamp": datetime.utcnow()
        })

        agent_msg_id = str(uuid.uuid4())
        db.collection("conversations").document(conv_id).collection("messages").document(agent_msg_id).set({
            "sender": "agent",
            "text": response_text,
            "timestamp": datetime.utcnow()
        })

        return {
            "reply": response_text,
            # "protocol": f"PROT-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "timestamp": datetime.now().isoformat()}