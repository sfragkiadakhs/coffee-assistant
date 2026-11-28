"""
    Coffee Assistant as an MCP server: any MCP client 
    (Claude Desktop, VS Code...) can ask it coffee questions.
"""

from mcp.server import MCPServer

from coffee_assistant import rag

mcp = MCPServer("Coffee Assistant")


@mcp.tool() # decorator that registers the function below as a tool
def ask_coffee_assistant(question: str) -> str:
    """
    Answer a coffee question (brewing, beans, caffeine, prices)
    from the Coffee Assistant database, with a web search fallback. 
    The answer names its source.
    """
    result = rag.rag(question)
    return result["answer"]

if __name__ == "__main__":
    mcp.run()  # no arguments = stdio: talk through standard input/output
