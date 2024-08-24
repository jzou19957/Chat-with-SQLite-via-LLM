import os
import glob
import sqlite3
import logging
import traceback
import subprocess
import sys
import pandas as pd
import numpy as np
import re
import json
import uuid
from typing import Dict, List, Any, Tuple
import google.generativeai as generativeai
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("data_analysis.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Configure Google Gemini API
API_KEY = "" #Replace this with your own free Gemni API
generativeai.configure(api_key=API_KEY)
model = generativeai.GenerativeModel('gemini-1.5-flash')

def print_and_log(message: str):
    """Utility function to print and log messages."""
    print(message)
    logger.info(message)

def call_generative_api(prompt: str) -> str:
    """Function to call the Google Gemini API with a prompt."""
    print_and_log("Sending request to Generative API...")
    try:
        response = model.generate_content(prompt)
        print_and_log("Received response from Generative API.")
        return response.text
    except Exception as exception:
        print_and_log(f"Error calling Generative API: {str(exception)}")
        return None

def extract_code_block(llm_response: str) -> str:
    """Extracts the code block from the LLM response."""
    code_match = re.search(r"# BEGIN CODE(.*?)# END CODE", llm_response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    else:
        raise ValueError("No valid code block found in the LLM response.")

def execute_python_code(code: str, output_folder: str) -> str:
    """Executes the provided Python code and returns the output."""
    cleaned_code = extract_code_block(code)
    code_filename = f"analysis_code_{uuid.uuid4().hex[:8]}.py"
    
    with open(code_filename, "w") as f:
        f.write(cleaned_code)

    try:
        result = subprocess.run([sys.executable, code_filename], capture_output=True, text=True, check=True)
        output_path = os.path.join(output_folder, f"{code_filename.replace('.py', '_output.txt')}")
        with open(output_path, "w") as output_file:
            output_file.write(result.stdout)
        print_and_log(f"Python code executed successfully. Output saved to {output_path}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Error executing Python code: {e.stderr}\n{traceback.format_exc()}"
        print_and_log(error_message)
        raise RuntimeError(error_message)
    finally:
        os.remove(code_filename)  # Clean up generated script file

class DataAnalysisTool:
    def __init__(self):
        self.db_file = self.find_db_file()
        self.db_structure = self.get_db_structure()
        self.sample_data = self.get_sample_data()
        self.output_folder = "analysis_results"
        os.makedirs(self.output_folder, exist_ok=True)

    def find_db_file(self) -> str:
        """Finds the first .db file in the current directory."""
        db_files = glob.glob("*.db")
        if not db_files:
            raise FileNotFoundError("No .db file found in the current directory.")
        return db_files[0]

    def get_db_structure(self) -> Dict[str, List[str]]:
        """Retrieves the structure of the SQLite database."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        structure = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            structure[table_name] = [col[1] for col in columns]
        conn.close()
        return structure

    def get_sample_data(self, sample_size: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieves sample data from each table in the database."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        sample_data = {}
        for table_name, columns in self.db_structure.items():
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {sample_size}")
            rows = cursor.fetchall()
            sample_data[table_name] = [dict(zip(columns, row)) for row in rows]
        conn.close()
        return sample_data

    def assess_query_complexity(self, user_query: str) -> Tuple[str, str]:
        """Assesses the complexity of the user's query."""
        prompt = f"""
        Analyze the following user query and database structure to determine the most appropriate method to answer it:
        User Query: "{user_query}"
        Database Structure: {json.dumps(self.db_structure, indent=2)}
        Sample Data: {json.dumps(self.sample_data, indent=2)}
        Determine which of the following methods is most appropriate:
        1. "SIMPLE_SQLITE" - If the query can be answered with a straightforward SQLite query.
        2. "COMPLEX_SQLITE" - If the query requires a more complex SQLite query (e.g., multiple joins, subqueries).
        3. "PYTHON_PANDAS" - If the query requires data manipulation or analysis that's better suited for Python and pandas.
        4. "PYTHON_VISUALIZATION" - If the query requires creating charts or visualizations.
        Please respond with the code inside # BEGIN CODE and # END CODE markers.
        """
        response = call_generative_api(prompt)
        if not response:
            return "PYTHON_PANDAS", "Failed to assess query complexity. Defaulting to Python with pandas."
        decision, explanation = response.split("\n", 1)
        return decision.strip(), explanation.strip()

    def generate_sqlite_query(self, user_query: str, complexity: str) -> str:
        """Generates a SQLite query based on the user's query and complexity."""
        prompt = f"""
        Generate a SQLite query to precisely answer this question: '{user_query}'
        Query Complexity: {complexity}
        Database structure: {json.dumps(self.db_structure, indent=2)}
        Sample Data: {json.dumps(self.sample_data, indent=2)}
        Guidelines:
        1. Use a {'simple' if complexity == 'SIMPLE_SQLITE' else 'complex'} SQLite query to answer the question.
        2. Ensure the query is efficient and doesn't read unnecessary data.
        3. Format the query for readability.
        4. Handle null values and potential zero values appropriately, especially for calculations like averages.
        5. Provide counts of total records, valid records, and records excluded due to null or invalid values.
        6. For average calculations, use NULLIF to avoid division by zero, e.g., AVG(NULLIF(column, 0))
        7. If the question is ambiguous, provide multiple queries to address different interpretations.
        8. Include comments explaining any assumptions or decisions made in crafting the query.
        Please respond with the code inside # BEGIN CODE and # END CODE markers.
        """
        return call_generative_api(prompt)

    def generate_python_code(self, user_query: str, complexity: str) -> str:
        """Generates Python code to analyze the data."""
        prompt = f"""
        Generate Python code to precisely answer this question: '{user_query}'
        Analysis Type: {complexity}
        Database structure: {json.dumps(self.db_structure, indent=2)}
        Sample Data: {json.dumps(self.sample_data, indent=2)}
        Guidelines:
        1. The .db file is in the current working directory. Use glob to find it.
        2. Use SQLite to query the database and pandas for data manipulation.
        3. If the analysis type is PYTHON_VISUALIZATION, create visualizations using matplotlib or seaborn.
        4. Handle potential errors such as missing data or unexpected data types.
        5. Optimize for performance by only reading necessary data from the database.
        6. For calculations like averages, exclude null values and potentially invalid values (e.g., zero for age).
        7. Provide counts of total records, valid records, and records excluded due to null or invalid values.
        8. If the question is ambiguous, provide multiple analyses to address different interpretations.
        9. Include comments explaining any assumptions or decisions made in the analysis.
        10. Save all generated visualizations and data outputs to the '{self.output_folder}' directory.
        11. Present the final output as a list of bullet points, starting each with a dash (-).
        Please respond with the code inside # BEGIN CODE and # END CODE markers.
        """
        return call_generative_api(prompt)

    def execute_sqlite_query(self, query: str) -> pd.DataFrame:
        """Executes a SQLite query and returns the result as a DataFrame."""
        conn = sqlite3.connect(self.db_file)
        try:
            result = pd.read_sql_query(query, conn)
            output_path = os.path.join(self.output_folder, "sqlite_query_result.csv")
            result.to_csv(output_path, index=False)
            print_and_log(f"Query result saved to {output_path}")
            return result
        except Exception as e:
            logger.error(f"Error executing SQLite query: {str(e)}")
            raise
        finally:
            conn.close()

    def analyze(self, user_query: str) -> str:
        """Analyzes the database based on the user's query."""
        complexity, explanation = self.assess_query_complexity(user_query)
        print_and_log(f"Query complexity assessment: {complexity}")
        print_and_log(f"Explanation: {explanation}")

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if complexity in ("SIMPLE_SQLITE", "COMPLEX_SQLITE"):
                    try:
                        query = self.generate_sqlite_query(user_query, complexity)
                        result = self.execute_sqlite_query(query)
                        result_str = result.to_string()
                    except Exception:
                        print_and_log("SQLite execution failed, falling back to Python.")
                        complexity = "PYTHON_PANDAS"  # Fallback to Python
                        code = self.generate_python_code(user_query, complexity)
                        result_str = execute_python_code(code, self.output_folder)
                else:
                    code = self.generate_python_code(user_query, complexity)
                    result_str = execute_python_code(code, self.output_folder)

                interpretation = self.interpret_result(user_query, result_str, complexity)
                output_file_path = os.path.join(self.output_folder, "interpretation.txt")
                with open(output_file_path, "w") as output_file:
                    output_file.write(interpretation)
                print_and_log(f"Interpretation saved to {output_file_path}")
                return interpretation
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_attempts - 1:
                    return f"Failed to execute analysis after {max_attempts} attempts. Error: {str(e)}"

    def interpret_result(self, user_query: str, result: str, complexity: str) -> str:
        """Interprets the result of the analysis."""
        prompt = f"""
        The user asked: "{user_query}"
        The analysis method used was: {complexity}
        The analysis produced this result:
        {result}
        Please provide a comprehensive response to the user's question based on this result. Include:
        1. A precise answer to the user's question, with context.
        2. The number of records that were excluded due to null or invalid values, if applicable.
        3. Any assumptions made about the data.
        4. The total number of records considered and the number of valid records used in the calculation.
        5. Any potential issues or limitations with the analysis.
        6. If the question could have multiple interpretations, provide alternative answers.
        7. Present the response as a list of concise bullet points, starting each point with a dash (-).
        8. Ensure all numerical results are rounded to two decimal places.
        9. Keep the response detailed but concise, focusing on the most relevant information.
        """
        return call_generative_api(prompt)

def main():
    try:
        data_tool = DataAnalysisTool()
        print(f"Database file: {data_tool.db_file}")
        print(f"Database structure:\n{json.dumps(data_tool.db_structure, indent=2)}")
        
        # Hardcoded user query
        user_query = "find a way to visualize those high short volume stock and put their name on top of it and show all the symbols depending on their shrot vol size put in different y axis etc"

        # Analyzing the hardcoded query
        result = data_tool.analyze(user_query)
        print("\nAnalysis Result:")
        print(result)
        print("\n" + "="*50)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()