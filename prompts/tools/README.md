# Tool Prompt Templates

Each tool has a dedicated prompt that helps the LLM understand when and how to use it.

## search_knowledge_base
Use this tool when the user asks about specific information that may be in indexed documents.

## execute_code
Use this tool when the user requests code execution or computation.
Always validate inputs before executing. Never execute destructive operations.

## web_search
Use this tool when the user needs current information not in the knowledge base.
Summarize results with source URLs.
