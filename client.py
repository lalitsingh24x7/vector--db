"""
client.py
---------
Python MCP client for Jupyter — connects to RQM MCP server over SSE.

Usage:
    from client import MCPClient

    client = MCPClient()

    # SQL query
    result = await client.execute_query("SELECT * FROM instrument_metadata WHERE region = 'US'")

    # Count
    result = await client.execute_query("SELECT COUNT(*) FROM mortgage_data WHERE issuer = 'Fannie Mae'")

    # Group by
    result = await client.execute_query("SELECT asset_class, COUNT(*) as count FROM instrument_metadata GROUP BY asset_class")

    # Join
    result = await client.execute_query(
        "SELECT i.ticker, m.security_id FROM instrument_metadata i JOIN mortgage_data m ON i.region = m.region"
    )

    # Semantic search
    result = await client.vector_search("Do we have ticker USLC0003?")

    # List tables
    result = await client.list_tables()

    # Get schema
    result = await client.get_schema("instrument_metadata")
"""

import json
from mcp.client.sse import sse_client
from mcp import ClientSession

SERVER_URL = "http://localhost:8080/sse"


class MCPClient:
    def __init__(self, url: str = SERVER_URL):
        self.url = url

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        async with sse_client(self.url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return json.loads(result.content[0].text)

    async def execute_query(self, sql: str) -> dict:
        return await self._call_tool("execute_query", {"sql": sql})

    async def vector_search(self, question: str, top_k: int = 5, source_table: str = None) -> dict:
        args = {"question": question, "top_k": top_k}
        if source_table:
            args["source_table"] = source_table
        return await self._call_tool("vector_search", args)

    async def list_tables(self) -> dict:
        return await self._call_tool("list_tables", {})

    async def get_schema(self, table: str) -> dict:
        return await self._call_tool("get_schema", {"table": table})