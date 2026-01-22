from google.adk.agents.llm_agent import Agent
from sqlalchemy import create_engine,text

def query_data(sql: str) -> str:
    """Execute SQL queries safely on the Postgres database."""
    engine = create_engine(
		"postgresql+psycopg2://admin:admin123@psql:5432/postgres"
	)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
        rows = result.fetchall()
        return [dict(r._mapping) for r in rows]
    
    except Exception as e:
        return f"Error: {str(e)}"
   
def list_columns(table) -> str:
    """Execute SQL queries query of columns name and data type in the table that need to search."""
    engine = create_engine(
		"postgresql+psycopg2://admin:admin123@psql:5432/postgres"
	)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT column_name,data_type FROM information_schema.columns WHERE table_schema = 'warehouse' AND table_name = '{table}' ORDER BY ordinal_position"))
        rows = result.fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception as e:
        return f"Error: {str(e)}"

def list_table_from_stock() -> str:
    """List all tables from stock schema, help for chatbot to know which table to query."""
    engine = create_engine(
		"postgresql+psycopg2://admin:admin123@psql:5432/postgres"
	)
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'warehouse' ORDER BY table_name"))
        rows = result.fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception as e:
        return f"Error: {str(e)}"

root_agent = Agent(
    name="sql_assistant",
    model="gemini-2.5-flash",  
    instruction="""
    You are an expert SQL agent designed to assist users in querying and analyzing data stored in a PostgreSQL database.
    To do this, you must use all available tools to execute SQL queries on the database. table: 
    - warehouse.warehouse_overview: Contains general information about companies.
    - warehouse.warehouse_events: Contains information about dividend payments to shareholders.
    - warehouse.warehouse_officers: Contains information about company executives.
    - warehouse.warehouse_shareholders: Contains information about major shareholders of the company.
    Follow these steps to assist the user effectively:
    ### Note: when user ask about shareholder, only list about top 5 shareholder of company.
    Step 1: Know the database schema by using the 'list_table_from_stock' and 'list_columns' tools to understand the structure of the data.
    Step 2: Based on the user's query, define what table and columns are relevant based on the schema information obtained in Step 1.
    Step 3: Formulate an logiccal SQL query to retrieve the necessary data from the database using the 'query_data' tool.
    SELECT ...
    FROM warehouse.<table_name>
    WHERE <conditions>;
    GROUP BY ... (optional) - related to aggregate functions like COUNT, SUM, AVG, etc.
    ORDER BY ... (optional)
    HAVING ... (optional)
    
    """,
    description="A helpful assistant to help with performing SQL queries on a Postgres database.",
    tools=[query_data,list_columns,list_table_from_stock],
)