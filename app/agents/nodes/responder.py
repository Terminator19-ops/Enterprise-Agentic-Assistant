import logfire
from app.agents.state import AgentState
from app.config import settings
from langchain_groq import ChatGroq

llm = ChatGroq(api_key=settings.GROQ_API_KEY, model= settings.GROQ_MODEL, temperature=0)

def generate_node(state: AgentState):
    """
    Synthesises a response using both
    Documentatuin context and conversational History.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}:{msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info('generating conversational responses using memory.')
        prompt = f"""
        You are an friendly and helpful AI assistant.
        Answer the latest user message using the CONVERSATION HISTORY below.
        
            CONVERSATION HISTORY: {history_str}
            LATEST MESSAGE: "{user_msg}" 
        """
    else:
        logfire.info("generating Technical RAG Responses")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context Truncated to fit Groq TPM limits.")
                break
        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using TECHNICAL CONTEXT provided

        TECHNICAL CONTEXT: {full_context}
        CONVERSATION HISTORY: {history_str}
        USER QUESTION: {user_msg}

        """    

    with logfire.span("LLM synthesis"):
        try: 
            content = llm.invoke(prompt).content
            logfire.info("response generated.")

            return{
                "final_answer": content,
                "status": "Response Generated",
                "plan": state["plan"],
                "messages":[{"role":"assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
