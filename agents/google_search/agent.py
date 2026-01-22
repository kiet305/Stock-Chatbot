from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

PROMPT = f"""
You are a helpful assistant that can search the web for information.
Current datetime: {current_time}

When you receive a query, you will perform a web search and return the most relevant results.
"""

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='google_search',
    description='A helpful assistant to help with performing Google searches and rendering the results.',
    instruction=PROMPT,
    tools=[google_search],
)
