import os
import sys
# Add the current directory to Python path to find src module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google.adk.agents import Agent
from src.a2a_tools.company_sql_search import SQLAgentCompany
from src.a2a_tools.simple_stock_eco import VectorDBStockAgent

from dotenv import load_dotenv
load_dotenv()

PROMPT = """
    You are customer service agent, master of all agents help user anwser related to stocks, finance, and economics, or company infomation. You are able to delegate tasks to the appropriate agents.
    You will be provided tools:
    - SQLAgentCompany: to help user find company information from company database.
    - VectorDBStockAgent: to help user find definitions and basic information about stocks, finance
"""

sql_company_search = SQLAgentCompany(agent_url=os.getenv("SQL_COMPANY_AGENT_URL"))
simple_stock_eco_agent = VectorDBStockAgent(agent_url=os.getenv("SIMPLE_STOCK_VECTOR_DB_AGENT_URL"))

root_agent = Agent(
    model='gemini-2.5-flash',
    name='a2a_agent_master',
    description='A helpful customer service assistant help delegate tasks to other agents related to stocks, finance, economics, and company information.',
    instruction=PROMPT,
    tools=[
        sql_company_search.invoke_company_sql_agent_via_a2a,
        simple_stock_eco_agent.invoke_vector_db_stock_agent_via_a2a
    ],
)
