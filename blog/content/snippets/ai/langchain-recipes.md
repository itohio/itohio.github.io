---
title: "LangChain Recipes"
date: 2024-12-13
draft: false
category: "ai"
tags: ["langchain", "llm", "ai", "chatgpt", "python"]
---

Practical recipes for building LLM applications with LangChain: prompts, chains, agents, memory, and RAG.

## Installation

```bash
# Core LangChain
pip install langchain langchain-community langchain-core

# OpenAI
pip install langchain-openai

# Other providers
pip install langchain-anthropic  # Claude
pip install langchain-google-genai  # Gemini

# Vector stores
pip install chromadb faiss-cpu  # or faiss-gpu

# Document loaders
pip install pypdf unstructured

# Utilities
pip install python-dotenv
```

## Setup

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
```

---

## Basic LLM Usage

### OpenAI

```python
from langchain_openai import ChatOpenAI, OpenAI

# Chat model (recommended)
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

response = llm.invoke("What is LangChain?")
print(response.content)

# Completion model (legacy)
llm = OpenAI(model="gpt-3.5-turbo-instruct")
response = llm.invoke("What is LangChain?")
```

### Anthropic Claude

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    temperature=0.7,
    max_tokens=1000
)

response = llm.invoke("What is LangChain?")
print(response.content)
```

### Streaming Responses

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", streaming=True)

for chunk in llm.stream("Write a short poem about AI"):
    print(chunk.content, end="", flush=True)
```

---

## Prompts and Templates

### Simple Prompt Template

```python
from langchain.prompts import PromptTemplate

# Create template
template = """
You are a helpful assistant. Answer the following question:

Question: {question}

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"]
)

# Format prompt
formatted_prompt = prompt.format(question="What is machine learning?")
print(formatted_prompt)

# Use with LLM
chain = prompt | llm
response = chain.invoke({"question": "What is machine learning?"})
```

### Chat Prompt Template

```python
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Create chat template
system_template = "You are a helpful assistant that {task}."
human_template = "{input}"

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

# Format
messages = chat_prompt.format_messages(
    task="translates English to French",
    input="Hello, how are you?"
)

# Use with LLM
response = llm.invoke(messages)
print(response.content)
```

### Few-Shot Prompting

```python
from langchain.prompts import FewShotPromptTemplate

# Examples
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "hot", "output": "cold"}
]

# Example template
example_template = """
Input: {input}
Output: {output}
"""

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template
)

# Few-shot template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of the word:",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)

# Use
chain = few_shot_prompt | llm
response = chain.invoke({"input": "big"})
```

---

## Chains

### Simple Sequential Chain

```python
from langchain.chains import LLMChain, SimpleSequentialChain

# First chain: generate topic
topic_template = "Suggest a topic about {subject}"
topic_prompt = PromptTemplate(
    input_variables=["subject"],
    template=topic_template
)
topic_chain = LLMChain(llm=llm, prompt=topic_prompt)

# Second chain: write about topic
write_template = "Write a short paragraph about: {topic}"
write_prompt = PromptTemplate(
    input_variables=["topic"],
    template=write_template
)
write_chain = LLMChain(llm=llm, prompt=write_prompt)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[topic_chain, write_chain],
    verbose=True
)

result = overall_chain.invoke("artificial intelligence")
print(result)
```

### Sequential Chain with Multiple Inputs/Outputs

```python
from langchain.chains import SequentialChain

# Chain 1: Translate
translate_template = "Translate this to {language}: {text}"
translate_prompt = PromptTemplate(
    input_variables=["language", "text"],
    template=translate_template
)
translate_chain = LLMChain(
    llm=llm,
    prompt=translate_prompt,
    output_key="translated_text"
)

# Chain 2: Summarize
summarize_template = "Summarize this text in one sentence: {translated_text}"
summarize_prompt = PromptTemplate(
    input_variables=["translated_text"],
    template=summarize_template
)
summarize_chain = LLMChain(
    llm=llm,
    prompt=summarize_prompt,
    output_key="summary"
)

# Combine
overall_chain = SequentialChain(
    chains=[translate_chain, summarize_chain],
    input_variables=["language", "text"],
    output_variables=["translated_text", "summary"],
    verbose=True
)

result = overall_chain.invoke({
    "language": "French",
    "text": "LangChain is a framework for developing applications powered by language models."
})

print("Translation:", result["translated_text"])
print("Summary:", result["summary"])
```

### Router Chain

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

# Define prompts for different topics
physics_template = """You are a physics expert. Answer this question:

{input}"""

math_template = """You are a math expert. Answer this question:

{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering physics questions",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    }
]

# Create destination chains
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt = PromptTemplate(
        template=p_info["prompt_template"],
        input_variables=["input"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# Default chain
default_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Answer this question: {input}",
        input_variables=["input"]
    )
)

# Router chain
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
router_template = f"""Given a user question, choose which expert should answer it.

Experts:
{chr(10).join(destinations)}

Question: {{input}}

Expert:"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"]
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# Multi-prompt chain
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

result = chain.invoke("What is Newton's second law?")
print(result)
```

---

## Memory

### Conversation Buffer Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Create memory
memory = ConversationBufferMemory()

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Have conversation
response1 = conversation.invoke("Hi, my name is Alice")
print(response1)

response2 = conversation.invoke("What's my name?")
print(response2)

# View conversation history
print(memory.load_memory_variables({}))
```

### Conversation Buffer Window Memory

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep only last k interactions
memory = ConversationBufferWindowMemory(k=2)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.invoke("Hi, I'm Alice")
conversation.invoke("I like pizza")
conversation.invoke("I have a cat")
conversation.invoke("What do I like?")  # Remembers pizza
conversation.invoke("What's my name?")  # Forgets Alice (outside window)
```

### Conversation Summary Memory

```python
from langchain.memory import ConversationSummaryMemory

# Summarize conversation history
memory = ConversationSummaryMemory(llm=llm)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.invoke("Hi, I'm working on a machine learning project")
conversation.invoke("I'm using Python and TensorFlow")
conversation.invoke("The model accuracy is 95%")

# View summary
print(memory.load_memory_variables({}))
```

### Conversation Summary Buffer Memory

```python
from langchain.memory import ConversationSummaryBufferMemory

# Hybrid: keep recent messages, summarize old ones
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100  # Summarize when exceeds limit
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
```

---

## Document Loading and Processing

### Load Documents

```python
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, DirectoryLoader,
    WebBaseLoader, CSVLoader
)

# Text file
loader = TextLoader("document.txt")
docs = loader.load()

# PDF
loader = PyPDFLoader("document.pdf")
pages = loader.load_and_split()

# Directory of files
loader = DirectoryLoader("./docs", glob="**/*.txt")
docs = loader.load()

# Web page
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# CSV
loader = CSVLoader("data.csv")
docs = loader.load()
```

### Text Splitting

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

# Recursive splitter (recommended)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(docs)

# Character splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)

# Token splitter
text_splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

---

## Vector Stores and Embeddings

### Create Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# HuggingFace embeddings (free, local)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Embed text
text = "This is a test document"
vector = embeddings.embed_query(text)
print(f"Vector dimension: {len(vector)}")
```

### Chroma Vector Store

```python
from langchain_community.vectorstores import Chroma

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Similarity search
query = "What is LangChain?"
docs = vectorstore.similarity_search(query, k=3)

for doc in docs:
    print(doc.page_content)
    print("---")

# Similarity search with scores
docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)

for doc, score in docs_with_scores:
    print(f"Score: {score}")
    print(doc.page_content)
    print("---")

# Load existing vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

### FAISS Vector Store

```python
from langchain_community.vectorstores import FAISS

# Create FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save
vectorstore.save_local("faiss_index")

# Load
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Search
docs = vectorstore.similarity_search(query, k=3)
```

---

## Retrieval-Augmented Generation (RAG)

### Basic RAG

```python
from langchain.chains import RetrievalQA

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff", "map_reduce", "refine", "map_rerank"
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# Ask question
result = qa_chain.invoke("What is LangChain?")
print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(doc.page_content[:200])
```

### Conversational RAG

```python
from langchain.chains import ConversationalRetrievalChain

# Create conversational chain with memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    ),
    verbose=True
)

# Have conversation
response1 = qa_chain.invoke({"question": "What is LangChain?"})
print(response1["answer"])

response2 = qa_chain.invoke({"question": "What are its main features?"})
print(response2["answer"])
```

### Custom RAG with LCEL

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Define prompt
template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Use
answer = rag_chain.invoke("What is LangChain?")
print(answer)
```

---

## Agents

### Basic Agent

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

# Create tools
search = DuckDuckGoSearchRun()

tools = [
    search
]

# Get prompt
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm, tools, prompt)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Run
result = agent_executor.invoke({
    "input": "What is the current weather in San Francisco?"
})
print(result["output"])
```

### Custom Tools

```python
from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

# Simple tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

multiply_tool = Tool(
    name="Multiply",
    func=multiply,
    description="Multiply two numbers together"
)

# Structured tool with validation
class CalculatorInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

def calculator(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

calculator_tool = StructuredTool.from_function(
    func=calculator,
    name="Calculator",
    description="Add two numbers",
    args_schema=CalculatorInput
)

# Use tools
tools = [multiply_tool, calculator_tool]
```

### Agent with Memory

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create agent with memory
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Have conversation
response1 = agent_executor.invoke({"input": "My name is Alice"})
response2 = agent_executor.invoke({"input": "What's my name?"})
```

---

## Output Parsers

### Structured Output

```python
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field

# Define schema
class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's occupation")

# Create parser
parser = PydanticOutputParser(pydantic_object=Person)

# Create prompt with format instructions
template = """Extract information about the person.

{format_instructions}

Text: {text}

Output:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create chain
chain = prompt | llm | parser

# Use
result = chain.invoke({
    "text": "John is a 30-year-old software engineer"
})

print(result.name)  # John
print(result.age)   # 30
print(result.occupation)  # software engineer
```

### JSON Output

```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Define schema
response_schemas = [
    ResponseSchema(name="answer", description="Answer to the question"),
    ResponseSchema(name="confidence", description="Confidence score 0-100")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create prompt
template = """Answer the question and provide confidence.

{format_instructions}

Question: {question}

Output:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Use
chain = prompt | llm | parser
result = chain.invoke({"question": "What is 2+2?"})

print(result["answer"])
print(result["confidence"])
```

---

## LangChain Expression Language (LCEL)

### Basic Chain

```python
from langchain_core.output_parsers import StrOutputParser

# Simple chain
chain = prompt | llm | StrOutputParser()

# Invoke
result = chain.invoke({"question": "What is AI?"})

# Batch
results = chain.batch([
    {"question": "What is AI?"},
    {"question": "What is ML?"}
])

# Stream
for chunk in chain.stream({"question": "What is AI?"}):
    print(chunk, end="", flush=True)
```

### Parallel Chains

```python
from langchain_core.runnables import RunnableParallel

# Run multiple chains in parallel
chain = RunnableParallel(
    summary=prompt1 | llm | StrOutputParser(),
    translation=prompt2 | llm | StrOutputParser()
)

result = chain.invoke({"text": "Some text"})
print("Summary:", result["summary"])
print("Translation:", result["translation"])
```

### Conditional Routing

```python
from langchain_core.runnables import RunnableBranch

# Route based on input
branch = RunnableBranch(
    (lambda x: "python" in x["question"].lower(), python_chain),
    (lambda x: "javascript" in x["question"].lower(), js_chain),
    default_chain
)

result = branch.invoke({"question": "How to use Python?"})
```

---

## Best Practices

### Error Handling

```python
from langchain.callbacks import get_openai_callback

try:
    with get_openai_callback() as cb:
        result = chain.invoke({"question": "What is AI?"})
        print(f"Tokens used: {cb.total_tokens}")
        print(f"Cost: ${cb.total_cost}")
except Exception as e:
    print(f"Error: {e}")
```

### Caching

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# In-memory cache
set_llm_cache(InMemoryCache())

# SQLite cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Now LLM calls are cached
response1 = llm.invoke("What is AI?")  # Makes API call
response2 = llm.invoke("What is AI?")  # Uses cache
```

### Callbacks

```python
from langchain.callbacks import StdOutCallbackHandler

# Add callback for debugging
chain = prompt | llm
result = chain.invoke(
    {"question": "What is AI?"},
    config={"callbacks": [StdOutCallbackHandler()]}
)
```

---

## Production Patterns

### Environment Configuration

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4"
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"

settings = Settings()

llm = ChatOpenAI(
    api_key=settings.openai_api_key,
    model=settings.model_name,
    temperature=settings.temperature
)
```

### Rate Limiting

```python
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import time

def rate_limited_invoke(chain, input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

### Logging

```python
import logging
from langchain.callbacks import FileCallbackHandler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File callback
handler = FileCallbackHandler("langchain.log")

# Use with chain
result = chain.invoke(
    {"question": "What is AI?"},
    config={"callbacks": [handler]}
)
```

---

## Further Reading

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)
- [LangSmith](https://smith.langchain.com/) - Debugging and monitoring
- [LangServe](https://github.com/langchain-ai/langserve) - Deploy LangChain apps

