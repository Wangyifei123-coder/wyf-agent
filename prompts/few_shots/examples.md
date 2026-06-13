# Few-Shot Examples

## Example 1: Knowledge Query
User: What is the architecture of the system?
Assistant:
THOUGHT: The user is asking about system architecture. I should search the knowledge base.
ACTION: search_knowledge_base
ACTION_INPUT: {"query": "system architecture"}
OBSERVATION: Found document "ARCHITECTURE.md" with relevant content about 7-layer architecture.
THOUGHT: I have the architecture information. Let me provide a clear answer.
ANSWER: The system uses a 7-layer architecture: LLM Gateway, Tools, Memory, Reasoning, Orchestration, Safety, and Observability. Each layer has a single responsibility and communicates through defined interfaces.

## Example 2: Code Task
User: Write a function to calculate fibonacci numbers.
Assistant:
THOUGHT: The user wants a fibonacci function. This is straightforward code generation.
ANSWER: Here's a Python implementation:

```python
def fibonacci(n: int) -> list[int]:
    if n <= 0:
        return []
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib[:n]
```

## Example 3: Multi-Step Task
User: Find all TODO comments in the codebase and summarize them.
Assistant:
THOUGHT: I need to search for TODO comments across the codebase. Let me use the search tool.
ACTION: search_knowledge_base
ACTION_INPUT: {"query": "TODO", "type": "code_comments"}
OBSERVATION: Found 5 TODO comments across 3 files.
THOUGHT: I have the results. Let me organize them.
ANSWER: Found 5 TODO items:
1. src/gateway/client.py:42 - Implement streaming response caching
2. src/memory/manager.py:15 - Add Chroma vector store integration
3. src/tools/registry.py:88 - Add tool versioning support
4. src/safety/guard.py:31 - Improve injection detection with ML model
5. src/api.py:67 - Add rate limiting middleware
