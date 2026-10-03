from app.agents.state import AgentState
from app.config import settings
from langchain_groq import ChatGroq
import logfire

llm = ChatGroq(api_key=settings.GROQ_API_KEY,model=settings.GROQ_MODEL, temperature=0)

def planner_node(state: AgentState):
    """
    the planner determines if a
    search is needed based on the ENTIRE conversation
    """

#  messages -> AI / Human / Tool / System                  
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role} : {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["message"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyse the conversational History and the latest user message.

    CONVERSATION HISTORY: {history}
    LATEST MESSAGE: "{user_msg}" 

    Task:
    1. If the latest message is a greeting(hi, hello) or a question that can be answered using ONLY the conversation history above (eg: whats my name) respond with CONVERSATIONAL
    2. If itsd a technical question about Kubernetes, Intel or Networking that requires a fresh documentation, output a refined search query.
    Output ONLY "CONVERSATIONAL" or the search query.
    """ 

    with logfire.span(" Planner Decision:"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory).....",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return {
        "current_query": decision,
        "status": f"Technical Research Needed. Searching for {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }

    