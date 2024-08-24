# Chat-with-SQLite-via-LLM
This script facilitates dynamic interaction with SQLite databases using a language model (LLM) from Google Gemini. 
This script facilitates dynamic interaction with SQLite databases using a language model (LLM) from Google Gemini. It incorporates multiple Python libraries, such as os, glob, sqlite3, logging, subprocess, pandas, numpy, re, and others, to perform various tasks, including file handling, database querying, logging, and data manipulation.

Key Components:
Logging Configuration:

The script sets up logging to log messages both to a file (data_analysis.log) and the console for better debugging and monitoring.
Google Gemini API Integration:

The script integrates with the Google Gemini API to utilize a language model (LLM) for generating SQLite queries or Python code based on user input.
Utility Functions:

print_and_log(message: str): A utility function that prints and logs messages.
call_generative_api(prompt: str) -> str: Sends prompts to the Google Gemini API and returns the generated response.
extract_code_block(llm_response: str) -> str: Extracts code blocks from the LLM response.
DataAnalysisTool Class:

This class provides methods to interact with SQLite databases, assess query complexity, generate appropriate queries or Python code, and interpret results.
Methods include:
find_db_file(): Finds the first .db file in the current directory.
get_db_structure(): Retrieves the structure of the SQLite database.
get_sample_data(sample_size: int = 5): Fetches sample data from each table in the database.
assess_query_complexity(user_query: str) -> Tuple[str, str]: Determines the complexity of the user's query to decide whether it requires a simple SQLite query, a complex SQLite query, or a Python script.
generate_sqlite_query(user_query: str, complexity: str) -> str: Generates a suitable SQLite query.
generate_python_code(user_query: str, complexity: str) -> str: Generates Python code for more complex analyses.
execute_sqlite_query(query: str) -> pd.DataFrame: Executes a generated SQLite query and returns the result as a DataFrame.
analyze(user_query: str) -> str: Orchestrates the analysis process based on the user's query.
interpret_result(user_query: str, result: str, complexity: str) -> str: Interprets the analysis results and provides a comprehensive response to the user's query.
Main Function:

The script’s entry point, which initializes the DataAnalysisTool, outputs the database structure, and analyzes a hardcoded user query to demonstrate its functionality.
Usage:
This script is designed for users who need to perform complex data analysis tasks on SQLite databases using natural language queries. The integration with an LLM allows it to dynamically generate and execute SQL queries or Python code to process and analyze the data, providing versatile and powerful data handling capabilities.

![05359ccf52ac2baffb69e50d90e84db](https://github.com/user-attachments/assets/a102c9c0-8a0a-48cf-b9ad-25a78a8e33fe)
![7c2318351594cd5f7b8dd8fd33ebdc9](https://github.com/user-attachments/assets/fb19041f-8ad5-42fc-938c-f0d84e0d41b4)
