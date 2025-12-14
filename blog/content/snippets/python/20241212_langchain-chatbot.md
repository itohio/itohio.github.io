---
title: "LangChain Chatbot with Tools"
date: 2024-12-12
draft: false
category: "python"
tags: ["python-knowhow", "langchain", "llm", "ai", "chatbot", "openrouter"]
---


Simple stdin chatbot using LangChain with tool calling (OpenRouter).

---

## Installation

```bash
pip install langchain langchain-openai python-dotenv
```

---

## Environment Setup

```bash
# .env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openai/gpt-4-turbo-preview
```

---

## Basic Chatbot

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# Initialize OpenRouter LLM
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4-turbo-preview"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)

def chat():
    """Simple chat loop"""
    messages = [
        SystemMessage(content="You are a helpful assistant.")
    ]
    
    print("Chatbot ready! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message
        messages.append(HumanMessage(content=user_input))
        
        # Get AI response
        response = llm.invoke(messages)
        messages.append(AIMessage(content=response.content))
        
        print(f"AI: {response.content}\n")

if __name__ == "__main__":
    chat()
```

---

## Chatbot with Tools

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

# Define tools
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Sum of a and b
    """
    return a + b

@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract b from a.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Difference of a and b
    """
    return a - b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Product of a and b
    """
    return a * b

@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide a by b.
    
    Args:
        a: Numerator
        b: Denominator
    
    Returns:
        Quotient of a divided by b
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4-turbo-preview"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

# Create tools list
tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers]

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful math assistant. Use the provided tools to help users with calculations."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def chat_with_tools():
    """Chat loop with tool support"""
    chat_history = []
    
    print("Math Chatbot ready! Type 'quit' to exit.")
    print("Try: 'What is 15 + 27?' or 'Calculate 100 divided by 4'\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            # Invoke agent
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Update history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response["output"]))
            
            print(f"AI: {response['output']}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    chat_with_tools()
```

---

## Advanced: Custom Tools

```python
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

@tool
def get_current_time() -> str:
    """Get the current time.
    
    Returns:
        Current time as a formatted string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate_age(birth_year: int) -> int:
    """Calculate age from birth year.
    
    Args:
        birth_year: Year of birth
    
    Returns:
        Current age
    """
    current_year = datetime.now().year
    return current_year - birth_year

@tool
def convert_temperature(temp: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin.
    
    Args:
        temp: Temperature value
        from_unit: Source unit (C, F, or K)
        to_unit: Target unit (C, F, or K)
    
    Returns:
        Converted temperature
    """
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    
    # Convert to Celsius first
    if from_unit == 'F':
        celsius = (temp - 32) * 5/9
    elif from_unit == 'K':
        celsius = temp - 273.15
    else:
        celsius = temp
    
    # Convert from Celsius to target
    if to_unit == 'F':
        return celsius * 9/5 + 32
    elif to_unit == 'K':
        return celsius + 273.15
    else:
        return celsius

@tool
def calculate_compound_interest(
    principal: float,
    rate: float,
    time: float,
    compounds_per_year: int = 1
) -> dict:
    """Calculate compound interest.
    
    Args:
        principal: Initial amount
        rate: Annual interest rate (as percentage, e.g., 5 for 5%)
        time: Time period in years
        compounds_per_year: Number of times interest is compounded per year
    
    Returns:
        Dictionary with final amount and interest earned
    """
    rate_decimal = rate / 100
    amount = principal * (1 + rate_decimal / compounds_per_year) ** (compounds_per_year * time)
    interest = amount - principal
    
    return {
        "final_amount": round(amount, 2),
        "interest_earned": round(interest, 2),
        "principal": principal,
        "rate": rate,
        "time": time
    }

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4-turbo-preview"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

# Create tools list
tools = [
    get_current_time,
    calculate_age,
    convert_temperature,
    calculate_compound_interest
]

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to various tools.
    Use the tools when needed to provide accurate information.
    Always explain your calculations and reasoning."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def advanced_chat():
    """Advanced chat with multiple tools"""
    chat_history = []
    
    print("Advanced Chatbot ready! Type 'quit' to exit.")
    print("Available capabilities:")
    print("- Get current time")
    print("- Calculate age from birth year")
    print("- Convert temperatures")
    print("- Calculate compound interest")
    print()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response["output"]))
            
            print(f"AI: {response['output']}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    advanced_chat()
```

---

## With Memory

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

load_dotenv()

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4-turbo-preview"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

tools = [add_numbers]

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember the conversation context."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

def chat_with_memory():
    """Chat with conversation memory"""
    print("Chatbot with memory ready! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"AI: {response['output']}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    chat_with_memory()
```

---

## Streaming Responses

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4-turbo-preview"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    streaming=True,
    temperature=0.7,
)

def chat_streaming():
    """Chat with streaming responses"""
    messages = [SystemMessage(content="You are a helpful assistant.")]
    
    print("Streaming Chatbot ready! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        messages.append(HumanMessage(content=user_input))
        
        print("AI: ", end="", flush=True)
        full_response = ""
        
        for chunk in llm.stream(messages):
            content = chunk.content
            print(content, end="", flush=True)
            full_response += content
        
        print("\n")
        messages.append(AIMessage(content=full_response))

if __name__ == "__main__":
    chat_streaming()
```

---

## Complete Example Script

```python
#!/usr/bin/env python3
"""
LangChain Chatbot with OpenRouter
Usage: python chatbot.py
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

# Validate environment
if not os.getenv("OPENROUTER_API_KEY"):
    print("Error: OPENROUTER_API_KEY not set in .env file")
    sys.exit(1)

# Tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Setup
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4-turbo-preview"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

tools = [add, subtract, multiply, divide]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful math assistant. Use tools for calculations."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

def main():
    """Main chat loop"""
    chat_history = []
    
    print("🤖 Math Chatbot (OpenRouter)")
    print("=" * 40)
    print("Commands: quit, exit, q")
    print("Example: 'What is 15 + 27?'")
    print("=" * 40)
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response["output"]))
            
            print(f"🤖: {response['output']}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()
```

---