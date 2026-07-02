# Goal
Build a multi-agent Python pipeline using LangChain to query the Semantic Scholar and Scite APIs.

# Target Data
Identify agriculture-related research papers. Focus specifically on methodologies utilizing computer vision for disease detection or time-series forecasting for crop yields. Prioritize papers detailing PyTorch or JAX implementations.

# Execution Rules
- Handle all pagination and API rate limits cleanly with exponential backoff.
- Output the final filtered dataset into a structured SQLite database.
