"""
HR Tools Module
Contains all HR agent functionality including LangGraph workflows and simple fallback agents.
"""

# Try to import LangGraph components
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from typing import TypedDict, Annotated, Sequence
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("⚠️  LangGraph not available. HR Agent will use simple LLM responses.")
    # Define dummy types for when LangGraph is not available
    BaseMessage = None
    add_messages = None

# HR Agent State (only used when LangGraph is available)
if HAS_LANGGRAPH:
    class HRState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        service: str
        context: dict
else:
    # Dummy class for when LangGraph is not available
    class HRState:
        pass


def create_hr_agent_graph(service_type, get_llm_func, llm_provider):
    """
    Create LangGraph workflow for HR agent based on service type.
    
    Args:
        service_type: Type of service (not currently used, but kept for compatibility)
        get_llm_func: Function to get LLM instance (from main.py)
        llm_provider: LLM provider name (from main.py)
    
    Returns:
        Compiled LangGraph workflow or None if LangGraph not available
    """
    if not HAS_LANGGRAPH:
        return None
    
    # Define workflow nodes
    def route_request(state: HRState):
        """Route to appropriate handler based on service - returns state unchanged"""
        return state
    
    def handle_leave(state: HRState):
        """Handle leave management requests"""
        messages = state["messages"]
        
        # Create specialized prompt for leave management
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an HR assistant specialized in Leave Management.
            You can help with:
            - Checking leave balance
            - Requesting leaves (sick, vacation, personal)
            - Viewing leave history
            - Explaining leave policies
            
            Be helpful, professional, and provide specific information when available.
            If you need employee ID or dates, ask for them."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        llm = get_llm_func(provider=llm_provider)
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        
        return {
            "messages": [AIMessage(content=response.content if hasattr(response, 'content') else str(response))]
        }
    
    def handle_payroll(state: HRState):
        """Handle payroll management requests"""
        messages = state["messages"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an HR assistant specialized in Payroll Management.
            You can help with:
            - Viewing payslips
            - Understanding salary breakdown
            - Tax information
            - Deductions and benefits
            - Pay schedule
            
            Be professional and provide clear explanations.
            If you need specific employee information, ask for employee ID."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        llm = get_llm_func(provider=llm_provider)
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        
        return {
            "messages": [AIMessage(content=response.content if hasattr(response, 'content') else str(response))]
        }
    
    def handle_recruitment(state: HRState):
        """Handle recruitment management requests"""
        messages = state["messages"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an HR assistant specialized in Recruitment Management.
            You can help with:
            - Posting job openings
            - Reviewing applications
            - Scheduling interviews
            - Candidate screening
            - Job descriptions
            
            Be professional and guide through the recruitment process.
            If you need job details or candidate information, ask for specifics."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        llm = get_llm_func(provider=llm_provider)
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        
        return {
            "messages": [AIMessage(content=response.content if hasattr(response, 'content') else str(response))]
        }
    
    def general_response(state: HRState):
        """General HR response"""
        messages = state["messages"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful HR assistant. Provide friendly and professional assistance."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        llm = get_llm_func(provider=llm_provider)
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        
        return {
            "messages": [AIMessage(content=response.content if hasattr(response, 'content') else str(response))]
        }
    
    # Router function for conditional edges
    def route_decision(state: HRState):
        """Route to appropriate handler based on service"""
        service = state.get("service", "")
        if service == "leave":
            return "handle_leave"
        elif service == "payroll":
            return "handle_payroll"
        elif service == "recruitment":
            return "handle_recruitment"
        return "general_response"
    
    # Build graph
    workflow = StateGraph(HRState)
    
    # Add nodes
    workflow.add_node("route", route_request)
    workflow.add_node("handle_leave", handle_leave)
    workflow.add_node("handle_payroll", handle_payroll)
    workflow.add_node("handle_recruitment", handle_recruitment)
    workflow.add_node("general_response", general_response)
    
    # Set entry point
    workflow.set_entry_point("route")
    
    # Set conditional edges from route
    workflow.add_conditional_edges(
        "route",
        route_decision,
        {
            "handle_leave": "handle_leave",
            "handle_payroll": "handle_payroll",
            "handle_recruitment": "handle_recruitment",
            "general_response": "general_response"
        }
    )
    
    # All handlers end
    workflow.add_edge("handle_leave", END)
    workflow.add_edge("handle_payroll", END)
    workflow.add_edge("handle_recruitment", END)
    workflow.add_edge("general_response", END)
    
    return workflow.compile()


def simple_hr_agent(service: str, message: str, get_llm_func, llm_provider):
    """
    Simple HR agent without LangGraph.
    
    Args:
        service: Service type (leave, payroll, recruitment)
        message: User message
        get_llm_func: Function to get LLM instance (from main.py)
        llm_provider: LLM provider name (from main.py)
    
    Returns:
        Response string from the LLM
    """
    service_prompts = {
        "leave": """You are an HR assistant specialized in Leave Management.
        You can help with:
        - Checking leave balance
        - Requesting leaves (sick, vacation, personal)
        - Viewing leave history
        - Explaining leave policies
        
        Be helpful, professional, and provide specific information when available.""",
        "payroll": """You are an HR assistant specialized in Payroll Management.
        You can help with:
        - Viewing payslips
        - Understanding salary breakdown
        - Tax information
        - Deductions and benefits
        - Pay schedule
        
        Be professional and provide clear explanations.""",
        "recruitment": """You are an HR assistant specialized in Recruitment Management.
        You can help with:
        - Posting job openings
        - Reviewing applications
        - Scheduling interviews
        - Candidate screening
        - Job descriptions
        
        Be professional and guide through the recruitment process."""
    }
    
    prompt_text = service_prompts.get(service, "You are a helpful HR assistant.")
    prompt = ChatPromptTemplate.from_template(f"{prompt_text}\n\nUser: {{message}}\n\nAssistant:")
    
    llm = get_llm_func(provider=llm_provider)
    response = llm.invoke(prompt.format(message=message))
    
    return response.content if hasattr(response, 'content') else str(response)

