import os
import sys
# Add the current directory to Python path to find src module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google.adk.agents import Agent
from src.a2a_tools.customer_service import CustomerServiceAgent
from src.a2a_tools.google_search import GoogleSearchAgent
from src.a2a_tools.market_search import MarketSearchAgent

from dotenv import load_dotenv
load_dotenv()

PROMPT = """
You are the Master Agent responsible for coordinating all other agents.
Your role is to analyze the user's request and delegate the task to the most appropriate specialized agent.

Workflow:
Step 1: Analyze the user's request and determine which specialized agent is best suited to handle it.
Step 2: Delegate the task by invoking the selected agent's tool using the required format.
Step 3: After receiving the response from the delegated agent, optionally enhance or verify the answer using GoogleSearchAgent if additional or up-to-date information is needed.
Step 4: Provide a clear, accurate, and user-friendly final response to the user.

Available agents:
- CustomerServiceAgent: Handles questions related to stocks, finance, economics, and company information.
- GoogleSearchAgent: Retrieves up-to-date or external information from the web when necessary.
Step 5: Always use GoogleSearchAgent to verify or enrich responses from other agents when the information might be outdated or incomplete.
Important rules:
- Always delegate tasks instead of answering directly.
- Only use GoogleSearchAgent when the delegated agent’s response requires verification or enrichment.
- Do not expose internal agent logic or tool calls to the user.
"""

customer_service_agent = CustomerServiceAgent(agent_url=os.getenv("CUSTOMER_SERVICE_AGENT_URL"))
google_search_agent = GoogleSearchAgent(agent_url=os.getenv("GOOGLE_SEARCH_AGENT_URL"))
market_search_agent = MarketSearchAgent(agent_url=os.getenv("MARKET_SEARCH_AGENT_URL"))

root_agent = Agent(
    model='gemini-2.5-flash',
    name='a2a_agent_master',
    description='A helpful assistant help delegate tasks to other agents',
    instruction=PROMPT,
    tools=[
        customer_service_agent.invoke_customer_service_agent_via_a2a,
        google_search_agent.invoke_google_search_agent_via_a2a,
        market_search_agent.invoke_market_search_agent_via_a2a,
    ],
)
