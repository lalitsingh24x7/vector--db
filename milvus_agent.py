#!/usr/bin/env python
# coding: utf-8
"""
Simple OpenAI Agent with Milvus vector search + MCP SQL queries.
"""

import json
import asyncio
from openai import OpenAI
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# Import MCP client
from client import MCPClient

# ══════════════════════════════════════════════════════════════
# 1. SETUP: Milvus + Embedder + MCP
# ══════════════════════════════════════════════════════════════

# Connect to Milvus
connections.connect(
    alias="default",
    host="localhost",
    port=19530,
    db_name="rqm_db"
)

# Load collection and embedder
collection = Collection("rqm_metadata")
collection.load()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# OpenAI client
openai_client = OpenAI()  # Uses OPENAI_API_KEY env var

# MCP client (for SQL queries)
mcp_client = MCPClient()

print("✓ Connected to Milvus, loaded embedder, and initialized MCP client")


# ══════════════════════════════════════════════════════════════
# 2. TOOL FUNCTIONS
# ══════════════════════════════════════════════════════════════

def vector_search(query: str, source_table: str = None, top_k: int = 5) -> list[dict]:
    """
    Semantic search in Milvus vector database.
    
    Args:
        query: Natural language question or search terms
        source_table: Optional filter - 'instrument_metadata' or 'mortgage_data'
        top_k: Number of results to return
    
    Returns:
        List of matching records with score, source, description, and extra_fields
    """
    query_vector = embedder.encode([query]).tolist()
    
    expr = f'source_table == "{source_table}"' if source_table else None
    
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        expr=expr,
        output_fields=["source_table", "extra_fields", "description"]
    )
    
    matches = []
    for hit in results[0]:
        matches.append({
            "score": round(hit.score, 4),
            "source_table": hit.entity.get("source_table"),
            "description": hit.entity.get("description"),
            "extra_fields": json.loads(hit.entity.get("extra_fields", "{}"))
        })
    
    return matches


def execute_sql_sync(sql_query: str) -> dict:
    """
    Execute SQL query via MCP client (sync wrapper).
    
    Args:
        sql_query: SQL query string
    
    Returns:
        Query results as dict
    """
    try:
        # Add LIMIT if not present (to prevent token overflow)
        sql_upper = sql_query.upper()
        if "LIMIT" not in sql_upper and sql_upper.strip().startswith("SELECT"):
            # Don't add LIMIT to COUNT/aggregate queries
            if "COUNT(" not in sql_upper and "SUM(" not in sql_upper and "AVG(" not in sql_upper:
                sql_query = sql_query.rstrip(";") + " LIMIT 20"
        
        result = asyncio.run(mcp_client.execute_query(sql_query))
        return truncate_result(result)
    except Exception as e:
        return {"error": str(e)}


def truncate_result(result: dict, max_records: int = 20, max_chars: int = 8000) -> dict:
    """
    Truncate large results to stay within token limits.
    """
    if isinstance(result, dict) and "error" in result:
        return result
    
    # If result is a list of records, truncate
    if isinstance(result, list):
        if len(result) > max_records:
            truncated = result[:max_records]
            return {
                "data": truncated,
                "truncated": True,
                "total_returned": len(result),
                "showing": max_records,
                "note": f"Showing first {max_records} of {len(result)} records. Use LIMIT or more specific filters."
            }
        return {"data": result, "count": len(result)}
    
    # If result is a dict with data
    if isinstance(result, dict):
        if "data" in result and isinstance(result["data"], list):
            if len(result["data"]) > max_records:
                result["data"] = result["data"][:max_records]
                result["truncated"] = True
                result["note"] = f"Showing first {max_records} records. Use LIMIT or more specific filters."
    
    # Final safety: truncate string representation
    result_str = json.dumps(result, default=str)
    if len(result_str) > max_chars:
        return {
            "error": "Result too large",
            "note": "Query returned too much data. Add LIMIT clause or use more specific filters.",
            "preview": result_str[:500] + "..."
        }
    
    return result


# ══════════════════════════════════════════════════════════════
# 3. OPENAI TOOLS DEFINITION
# ══════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Search for financial instruments or mortgage securities using semantic similarity. Use this to find records by description, ticker, asset class, region, issuer, CUSIP, etc. Returns matching records with similarity scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query, e.g., 'US equity instruments', 'ticker USLC0003', 'fixed rate MBS securities', 'Fannie Mae mortgages'"
                    },
                    "source_table": {
                        "type": "string",
                        "enum": ["instrument_metadata", "mortgage_data"],
                        "description": "Optional: filter by data source. 'instrument_metadata' for tickers/benchmarks, 'mortgage_data' for MBS/mortgage securities"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 20)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute SQL queries against the database via MCP. Use for exact matches, aggregations (COUNT, SUM, AVG), GROUP BY, and complex filtering. IMPORTANT: Always use LIMIT (max 20) for SELECT * queries to avoid large results. For counts, use COUNT(*) instead of retrieving all records. Tables: instrument_metadata (ticker, benchmark_id, asset_class, region, currency, start_date, end_date), mortgage_data (security_id, cusip, collateral_type, issuer, coupon, origination_year, maturity_date, region, currency, start_date, end_date)",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "SQL query to execute. Always include LIMIT for SELECT queries. E.g., 'SELECT * FROM instrument_metadata WHERE region = \"US\" LIMIT 10' or 'SELECT COUNT(*) FROM instrument_metadata WHERE region = \"US\"'"
                    }
                },
                "required": ["sql_query"]
            }
        }
    }
]


# ══════════════════════════════════════════════════════════════
# 4. AGENT LOOP
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a financial data assistant with access to two tools:

**1. vector_search** - Semantic similarity search in Milvus:
   - instrument_metadata: tickers, benchmark IDs, asset classes, regions, currencies
   - mortgage_data: MBS securities, CUSIPs, issuers, collateral types, coupons
   - Use for: finding similar records, fuzzy lookups, natural language queries

**2. execute_sql** - SQL queries via MCP:
   - Same tables with structured columns
   - Use for: exact counts (COUNT), aggregations (SUM, AVG), GROUP BY, exact filters

**IMPORTANT Guidelines:**
- For "show me all X" queries: use COUNT(*) first, then SELECT with LIMIT 10-20
- ALWAYS add LIMIT clause to SELECT queries (max 20 rows)
- Use vector_search for similarity/semantic queries: "find US equities", "ticker like USLC"
- Use execute_sql for exact queries: "how many instruments in US", "count by asset class"
- Present results clearly with key fields highlighted
- If one tool fails, try the other as fallback
"""


def run_agent(user_message: str, conversation_history: list = None) -> str:
    """
    Run the agent with a user message.
    
    Args:
        user_message: User's question
        conversation_history: Optional list of previous messages
    
    Returns:
        Agent's response
    """
    if conversation_history is None:
        conversation_history = []
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": user_message}
    ]
    
    # Initial call
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )
    
    assistant_message = response.choices[0].message
    
    # Handle tool calls (loop until no more tools needed)
    while assistant_message.tool_calls:
        messages.append(assistant_message)
        
        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            print(f"🔧 Calling {func_name}({func_args})")
            
            # Execute the tool
            if func_name == "vector_search":
                result = vector_search(**func_args)
            elif func_name == "execute_sql":
                result = execute_sql_sync(func_args["sql_query"])
            else:
                result = {"error": f"Unknown function: {func_name}"}
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str)
            })
        
        # Get next response
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        assistant_message = response.choices[0].message
    
    return assistant_message.content


# ══════════════════════════════════════════════════════════════
# 5. INTERACTIVE CHAT
# ══════════════════════════════════════════════════════════════

def chat():
    """Interactive chat loop."""
    print("=" * 60)
    print("🤖 Financial Data Agent")
    print("   Tools: Vector Search (Milvus) + SQL (MCP)")
    print("   Type 'quit' to exit")
    print("=" * 60)
    
    history = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        try:
            response = run_agent(user_input, history)
            print(f"\n🤖 Agent: {response}")
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            # Keep history manageable
            if len(history) > 20:
                history = history[-20:]
                
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ══════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example queries to test
    print("\n📋 Example queries you can try:")
    print("  - Find US equity instruments")
    print("  - Do we have ticker USLC0003?")
    print("  - How many instruments are in the US region?")
    print("  - Count instruments by asset class")
    print("  - Show me fixed rate MBS securities")
    print("  - Find mortgage securities issued by Fannie Mae")
    print()
    
    chat()
