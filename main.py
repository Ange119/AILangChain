import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# LangChain detectará automáticamente la variable OPENAI_API_KEY del archivo .env, 
# pero si quieres asegurarte o usarla explícitamente:
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model = "gpt-4o-mini")