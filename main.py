import firebase_admin
from firebase_admin import credentials, firestore
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agents.agent import ChatbotAgent
import uuid
from datetime import datetime

cred = credentials.Certificate("FIREBASE_CREDENTIALS")
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

class Message(BaseModel):
    text: str
    user_id: str
    msg_id: str

class EndMessage(BaseModel):  
    user_id: str
    msg_id: str

@app.get("/")
def root():
    return {"message": "LuxCare AI API está rodando!"}

@app.post("/chat")
def chat(message: Message):
    try:
        user_history = agent.get_or_create_history(message.user_id)
        user_history.append({"role": "user", "content": message.text})
        
        state = {
            "messages": user_history,  
            "user_id": message.user_id
        }
        
        result = agent.graph.invoke(state)
        
        new_messages = result["messages"][len(user_history):]  
        for msg in new_messages:
            user_history.append(msg)
        
        last_assistant = None
        for msg in reversed(user_history):
            if hasattr(msg, 'type') and msg.type == "ai":
                last_assistant = msg
                break
            elif isinstance(msg, dict) and msg.get("role") == "assistant":
                last_assistant = msg
                break
        
        if not last_assistant:
            raise HTTPException(status_code=500, detail="No assistant response")
        
        response = last_assistant.content if hasattr(last_assistant, 'content') else last_assistant["content"]

        user_id = message.user_id
        msg_id = message.msg_id  
        line = str(uuid.uuid4()) 

        db.collection("conversations").document(user_id).collection("chat").document(msg_id).collection("messages").document(line).set({
            "user_message": {
                "sender": "user",
                "text": message.text,
                "timestamp": datetime.utcnow()
            },
            "agent_message": {
                "sender": "agent", 
                "text": response, 
                "timestamp": datetime.utcnow()
            },
            "conversation_timestamp": datetime.utcnow()
        })

        return {
            "reply": response,
            "msg_id": msg_id,
            "user_id": message.user_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")

@app.post("/chat/end")
def end_chat(user_id: str, msg_id: str):
    try:
        if user_id in agent.conversation_history:
            history = agent.conversation_history[user_id]
            
            result = agent.invoke_end(history)
            
            db.collection("conversations").document(user_id).collection("resumos").document(msg_id).set({ 
                "resumo_atendimento": result.content,  
                "timestamp": datetime.utcnow(),
                "total_mensagens": len(history)
            })
            
            agent.clear_history(user_id)
        
        return {"status": "conversa encerrada", "user_id": user_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
