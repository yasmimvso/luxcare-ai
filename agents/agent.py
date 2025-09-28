from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
import os

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None 


class ChatbotAgent:

    def __init__(self):

        self.llm = init_chat_model(
            "gemini-flash-latest",          
            model_provider="google_genai",
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )

        self.conversation_history = {}

        graph_builder = StateGraph(State)

        graph_builder.add_node("chatbot", self.chatbot_conf)
        graph_builder.add_node("save_to_firebase", self.save_node)

        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", "save_to_firebase")
        graph_builder.add_edge("save_to_firebase", END)

        self.graph = graph_builder.compile()
    
    def chatbot_conf(self, state: State):
        last_message = state["messages"][-1]  

        messages = [
            {
                "role": "system",
                "content": """
                Você é um assistente virtual acolhedor e empático, cujo objetivo é coletar informações de triagem de saúde de forma clara, calma, profissional e humanizada.

                Comportamento e Tom:
                - Seja acolhedor, empático, paciente e calmo.
                - Use linguagem clara, simples e direta.
                - Evite jargões de saúde ou termos técnicos desnecessários.
                - Mantenha tom humanizado, mas não excessivamente informal.
                - Transmita segurança para que o usuário se sinta confortável em compartilhar informações.

                [PROTOCOLO DE EMERGÊNCIA - PRIORIDADE MÁXIMA]
                - Se o usuário mencionar sintomas graves ou sinais de alerta como:
                * dor no peito
                * falta de ar
                * desmaio
                * sangramento intenso
                * convulsão
                * perda de consciência
                * sintomas que possam ser interpretados como fatais
                - O agente deve IMEDIATAMENTE interromper a triagem e responder:

                "Entendi. Seus sintomas podem indicar uma situação de emergência. 
                Por favor, procure o pronto-socorro mais próximo ou ligue para o 192 imediatamente."

                - Após emitir esse aviso, o agente NÃO deve continuar a coleta de dados.

                Identidade do Agente:
                - Apresente-se no início da conversa como um assistente virtual que não substitui um profissional de saúde.
                - Explique que seu papel é apenas coletar informações para agilizar a consulta.

                Restrições:
                - NUNCA ofereça diagnósticos.
                - NUNCA sugira tratamentos, medicamentos ou dosagens.
                - Se o usuário perguntar sobre diagnóstico ou tratamento, responda educadamente que apenas um profissional de saúde pode fornecer essa orientação.

                Dados a serem coletados na triagem:
                - Queixa Principal
                - Sintomas Detalhados
                - Duração e Frequência
                - Intensidade (escala 0 a 10)
                - Histórico Relevante
                - Medidas Tomadas

                Armazenamento:
                - Ao final, produza um resumo estruturado da triagem.

                Objetivo final:
                - Conduzir a conversa de forma natural, garantindo que todas as informações necessárias sejam coletadas.
                - No fim, confirme com o usuário se o resumo da triagem está correto antes de encerrar.
                """
            }
        ] + state["messages"]  

        reply = self.llm.invoke(messages)

        return {"messages": state["messages"] + [{"role": "assistant", "content": reply.content}]}

    def save_node(self, state: State):
        """Aqui você pode salvar no Firebase (ainda vamos plugar no main)"""
        # Aqui ainda não temos o firestore, mas você pode retornar o state normal
        return state
   
    def invoke(self, message: str, user_id: str = "default"):

        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
    
        self.conversation_history[user_id].append({"role": "user", "content": message})
        
        state = {"messages": self.conversation_history[user_id], "user_id": user_id}
        result = self.graph.invoke(state)
        
        new_assistant_msg = result["messages"][-1]  
        self.conversation_history[user_id].append(new_assistant_msg)
        
        return new_assistant_msg["content"]

