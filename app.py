import os
import glob
import sqlite3
import logging
import subprocess
import sys
import pandas as pd
import re
import json
import importlib
import base64
from typing import Dict, List, Any, Tuple
from datetime import datetime
import google.generativeai as generativeai
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("data_analysis.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load configuration
CONFIG = {
    "API_KEY": os.getenv("GENERATIVE_API_KEY", "####M"),  # Replace with your own free API at https://ai.google.dev/
}

# Configure Google Gemini API
def configure_api():
    generativeai.configure(api_key=CONFIG["API_KEY"])
    return generativeai.GenerativeModel('gemini-1.5-flash')

model = configure_api()

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

def extract_code_blocks(llm_response: str) -> List[str]:
    """Extracts multiple code blocks from the LLM response."""
    code_blocks = re.findall(r"#begin(.*?)#end", llm_response, re.DOTALL)
    if not code_blocks:
        # Fallback to a more lenient extraction method if no code blocks are found
        code_blocks = re.findall(r"```python(.*?)```", llm_response, re.DOTALL)
        if not code_blocks:
            raise ValueError("No valid code blocks found in the LLM response.")
    return [block.strip() for block in code_blocks]

def clean_python_code(code: str) -> str:
    """Cleans the generated Python code to ensure it is valid."""
    code = re.sub(r'-\s*\*\*.*?\*\*\s*:', '', code)
    return code

def install_package(package_name: str):
    """Installs a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print_and_log(f"Successfully installed {package_name}.")
    except subprocess.CalledProcessError as e:
        print_and_log(f"Failed to install {package_name}: {e}")
        return False
    return True

def try_install_alternative(libraries: List[str]) -> str:
    """Tries to install alternative libraries until one succeeds."""
    for library in libraries:
        if install_package(library):
            return library
    raise RuntimeError(f"All library installation attempts failed: {', '.join(libraries)}")

def reflect_and_retry(code: str, error_message: str) -> str:
    """Generates a reflection prompt based on the error and previous code, and tries again."""
    reflection_prompt = f"""
    # The following Python code was executed, but an error occurred:
    
{code}

    # Error Message: "{error_message}"

    # Analyze the error in the context of SQLite database interactions or visualizations. Identify the cause of the error and whether it can be fixed easily. If an alternative approach is required, suggest it while maintaining continuity with the existing data files and visualizations. Any generated outputs like CSV files or images must be referenced appropriately in the revised code.

    # Generate a new Python code solution using a step-by-step approach to solve the problem systematically. Ensure each step logically follows from the previous one, reuses any existing outputs effectively, and includes testing where appropriate. Respond with the revised Python code inside #begin and #end markers.
    """
    new_code_response = call_generative_api(reflection_prompt)
    return new_code_response

def execute_python_code(code_blocks: List[str], output_folder: str) -> str:
    """Executes the provided Python code blocks sequentially and returns the output."""
    output_str = ""
    for idx, code in enumerate(code_blocks):
        cleaned_code = clean_python_code(code)
        code_filename = f"analysis_code_{idx}.py"
        
        with open(code_filename, "w") as f:
            f.write(cleaned_code)

        try:
            required_packages = re.findall(r"import (\w+)|from (\w+) import", cleaned_code)
            for package in required_packages:
                package_name = package[0] or package[1]
                try:
                    importlib.import_module(package_name)
                except ImportError:
                    alternatives = {
                        "matplotlib": ["plotly", "bokeh"],
                        "seaborn": ["plotly", "bokeh"],
                    }
                    package_to_install = try_install_alternative(alternatives.get(package_name, [package_name]))
                    cleaned_code = cleaned_code.replace(package_name, package_to_install)

            result = subprocess.run([sys.executable, code_filename], capture_output=True, text=True, check=True)
            output_path = os.path.join(output_folder, f"output_{idx}.txt")
            with open(output_path, "w") as output_file:
                output_file.write(result.stdout)
            print_and_log(f"Python code block {idx + 1} executed successfully. Output saved to {output_path}")
            output_str += result.stdout + "\n"
        except subprocess.CalledProcessError as e:
            error_message = e.stderr
            print_and_log(f"Error executing Python code block {idx + 1}: {error_message}")
            try:
                new_code_response = reflect_and_retry(cleaned_code, error_message)
                new_code_blocks = extract_code_blocks(new_code_response)
                return execute_python_code(new_code_blocks, output_folder)
            except Exception as reflection_exception:
                print_and_log(f"Reflection and retry failed: {reflection_exception}")
                raise RuntimeError(f"Error after reflection and retry: {reflection_exception}")
        finally:
            os.remove(code_filename)
    return output_str.strip()

def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class DataAnalysisTool:
    def __init__(self, db_file: str, user_query: str):
        self.db_file = db_file
        self.user_query = user_query
        self.db_structure = self.get_db_structure()
        self.sample_data = self.get_sample_data()
        self.output_folder = self.create_output_folder()
        os.makedirs(self.output_folder, exist_ok=True)

    def create_output_folder(self) -> str:
        """Creates a unique output folder based on the database file name and the user query."""
        base_name = os.path.splitext(self.db_file)[0]
        folder_name = f"{base_name}_Q_{self.user_query[:30].replace(' ', '_').replace(':', '')}"
        os.makedirs(folder_name, exist_ok=True)
        return folder_name

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

    def preprocess_user_query(self, raw_query: str) -> str:
        """Preprocesses the user's raw query using the LLM to generate a cleaned, specific, and enhanced query."""
        prompt = f"""
        You are tasked with preprocessing a user's raw query to make it suitable for analysis. The raw query is: '{raw_query}'
        
        Guidelines:
        1. Identify and correct any typos in the query.
        """
        cleaned_query = call_generative_api(prompt)
        return cleaned_query.strip() if cleaned_query else raw_query

    def assess_query_complexity(self, user_query: str) -> Tuple[str, str]:
        """Assesses the complexity of the user's query."""
        prompt = f"""
        Analyze the following user query and database structure to determine the most appropriate method to answer it:
        User Query: "{user_query}"
        Database Structure: {json.dumps(self.db_structure, indent=2)}
        Sample Data: {json.dumps(self.sample_data, indent=2)}
        Determine which of the following methods is most appropriate:
        1. "SIMPLE_SQLITE" - If the query can be answered with a straightforward SQLite query.
        2. "COMPLEX_SQLITE" - If the query requires a more complex SQLite query (e.g., multiple joins, subqueries, complex, multi-stage calculation).
        3. "PYTHON_PANDAS" - If the query requires data manipulation or analysis that's better suited for Python and pandas.
        4. "PYTHON_VISUALIZATION" - If the query requires creating charts or visualizations.
        Please respond with the code inside #begin and #end markers.
        """
        response = call_generative_api(prompt)
        if not response:
            return "PYTHON_PANDAS", "Failed to assess query complexity. Defaulting to Python with pandas."
        decision, explanation = response.split("\n", 1)
        return decision.strip(), explanation.strip()

    def generate_sqlite_query(self, user_query: str, complexity: str) -> str:
        """Generates a SQLite query based on the user's query and complexity."""
        initial_query_prompt = f"""
        Generate a SQLite query to precisely answer this question: '{user_query}'
        Query Complexity: {complexity}
        Database structure: {json.dumps(self.db_structure, indent=2)}
        Sample Data: {json.dumps(self.sample_data, indent=2)}
        Guidelines:
        0. You are the world's best SQLite expert, capable of generating precise, efficient, and well-optimized queries. - Adapt the query based on the specified complexity level to meet user requirements effectively. - Ensure the query is free from common errors, optimized for performance, and formatted for clarity and readability.
        1. Use a {'simple' if complexity == 'SIMPLE_SQLITE' else 'complex'} SQLite query to answer the question.
        2.5. Before you generate query, use common sense to see if there are errors and ambiguity in the query, fix and reformatted into good query first.
        2. Ensure the query is efficient and doesn't read unnecessary data.
        3. Format the query for readability.
        4. Include comments explaining any assumptions or decisions made in crafting the query.
        5. Ensure the query adheres to the guidelines provided.
        6. Handle potential errors such as missing data or unexpected data types.
        7. Optimize for performance by only reading necessary data from the database.
        8. Include comments explaining any assumptions or decisions made in crafting the query.
        Please respond with the code inside #begin and #end markers.
        """
        initial_query = call_generative_api(initial_query_prompt)
        if not initial_query:
            return None

        feedback_prompt = f"""
        You are an objective third-party evaluator tasked with reviewing the following SQLite query. ### Your Expertise: - You specialize in detecting errors, inefficiencies, and areas for improvement, including subtle mistakes or commonly overlooked imperfections. - You excel at providing direct, actionable, and immediately useful suggestions that can be easily applied by the programmer without causing conflicts. Please review the following SQLite query generated to answer the user's query: '{user_query}'
        Query:
        {initial_query}
        Provide feedback on the following aspects:
        1. Query efficiency and performance.
        2. Handling of potential errors and edge cases.
        3. Clarity and readability of the query.
        4. Adherence to the guidelines provided.
        5. Any other improvements or suggestions.
        Please respond with a detailed feedback report.
        """
        feedback = call_generative_api(feedback_prompt)
        if not feedback:
            return initial_query

        improved_query_prompt = f"""
        Based on the feedback provided, please revise the initial SQLite query to address the issues and improve the query.
        Feedback:
        {feedback}
        Initial Query:
        {initial_query}
        Please respond with the revised query inside #begin and #end markers.
        """
        improved_query = call_generative_api(improved_query_prompt)
        return improved_query

    def generate_python_code(self, user_query: str, complexity: str) -> str:
        """Generates Python code to analyze the data based on a user query and complexity level."""
        
        # Build the code generation prompt
        initial_code_prompt = f"""
        Generate Python code to answer the following question: '{user_query}'
        
        Analysis Type: {complexity}
        Database Structure:
        {json.dumps(self.db_structure, indent=2)}
        
        Sample Data:
        {json.dumps(self.sample_data, indent=2)}
        
        Guidelines for Code Generation:
        0. Think in a step by step method, in order to answer the question, what do I need to know first? How can I know? Plan for this and then step by step gather the calculated results then write a visualization code for that. Ideally all should be done in one code, but multiple code is fine as long as you remember that any temporary notes or files you generated you have to remember what it is called so you are always fetching the right content.  
        1. You are a Python expert known for writing efficient, well-organized, and error-free code.
        - Specialize in data visualization and analysis using SQLite databases.
        - Utilize Python libraries like `sqlite3` (for database interaction), `pandas` (for data manipulation), 
            and `matplotlib` or `seaborn` (for visualization).
        - Ensure the code is modular, easy to understand, with clear variable names and informative comments.
        - Optimize performance, handle edge cases, and anticipate input errors.
   
        2. The .db file is located in the current working directory. Use `glob` to locate it.
        - Only use this .db file; do not create new ones.
        
        3. When the user query is vague, proactively convert it into a suitable professional visualization. Sometimes the query
        can have very obvious mistakes which requires direct intervention (.e.g what is APPL symbol, when in fact it is a typo for "AAPL"). Always prioritize the database you have in terms of what it can help the query with. Do not create or generate anything which the database itself does not have or cannot be computed to have. Create no more than two visualization by default unless otherwise specified.
        
        4. For complex tasks requiring multi-step solutions:
        - Treat the problem like a multi-step math problem, with each step building on the previous one.
        - Figure out what you need to calculate first and what data should be processed into what before visualization attempt. Not all data are provided directly, some requires computation first before visualization.
        - Generate comprehensive code that addresses all steps to reach the final solution.
        
        5. Default to academic-style, publication-ready statistics and visualizations.
        - Make visualization choices proactively unless specified by the user.
        - Make sure the title, legends, color, font size, text location, are all proper. 
        
        6. Use SQLite for database queries and pandas for data manipulation.
        
        7. If the analysis type is PYTHON_VISUALIZATION:
        - Create visualizations using `matplotlib`, `seaborn`, or other suitable libraries.

        7.5. By default generate no more than two visualization and one statistic report each time. If the generalized visualization 
        contains no data, that means it has failed and should be scraped. 

        7.6 By default do not create new .db file or csv files or any separate data files, work directly with the provided .db file only. If any new .db file is created by the code by accident, delete it so that you stick strictly with the provided .db file. 
        
        8. Address potential errors such as missing or unexpected data types.
        
        9. Optimize performance by reading only necessary data from the database.
        
        10. For calculations like averages:
            - Exclude null and potentially invalid values (e.g., zero for age).
        
        11. Provide counts of total records, valid records, and records excluded due to null or invalid values.
        
        12. Include comments explaining assumptions or decisions made during the analysis.
        
        13. Save all generated visualizations and data outputs to the '{self.output_folder}' directory.
        
        Respond with the code enclosed within #begin and #end markers.
        """


        initial_code = call_generative_api(initial_code_prompt)
        if not initial_code:
            return None

        feedback_prompt = f"""
        You are an objective third-party evaluator. Please review the following Python code generated to answer the user's query: '{user_query}'
        Code:
        {initial_code}
        Provide feedback on the following aspects:
        1. Code efficiency and performance.
        2. Handling of potential errors and edge cases.
        3. Clarity and readability of the code.
        4. Adherence to the guidelines provided.
        5. Any other improvements or suggestions.
        Please respond with a detailed feedback report.
        """
        feedback = call_generative_api(feedback_prompt)
        if not feedback:
            return initial_code

        improved_code_prompt = f"""
        Based on the feedback provided, please revise the initial Python code to address the issues and improve the code.
        Feedback:
        {feedback}
        Initial Code:
        {initial_code}
        Please respond with the revised code inside #begin and #end markers.
        """
        improved_code = call_generative_api(improved_code_prompt)
        return improved_code

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
        cleaned_query = self.preprocess_user_query(user_query)
        complexity, explanation = self.assess_query_complexity(cleaned_query)
        print_and_log(f"Query complexity assessment: {complexity}")
        print_and_log(f"Explanation: {explanation}")

        analysis_metadata = {
            "user_query": user_query,
            "cleaned_query": cleaned_query,
            "db_file": self.db_file,
            "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result_description": "",
            "visualization_description": "",
            "visualization_base64": ""
        }

        max_attempts = 15
        for attempt in range(max_attempts):
            try:
                if complexity in ("SIMPLE_SQLITE", "COMPLEX_SQLITE"):
                    try:
                        query = self.generate_sqlite_query(cleaned_query, complexity)
                        result = self.execute_sqlite_query(query)
                        result_str = result.to_string()
                        analysis_metadata["result_description"] = "SQL Query executed successfully. Results saved in CSV format."
                    except Exception:
                        print_and_log("SQLite execution failed, falling back to Python.")
                        complexity = "PYTHON_PANDAS"  # Fallback to Python
                        code = self.generate_python_code(cleaned_query, complexity)
                        result_str = execute_python_code(extract_code_blocks(code), self.output_folder)
                        analysis_metadata["result_description"] = "Python analysis executed successfully."
                else:
                    code = self.generate_python_code(cleaned_query, complexity)
                    result_str = execute_python_code(extract_code_blocks(code), self.output_folder)
                    analysis_metadata["result_description"] = "Python visualization executed successfully."

                if complexity == "PYTHON_VISUALIZATION":
                    visualization_desc = self.describe_visualization(self.output_folder)
                    analysis_metadata["visualization_description"] = visualization_desc
                    for file in os.listdir(self.output_folder):
                        if file.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                            image_path = os.path.join(self.output_folder, file)
                            analysis_metadata["visualization_base64"] = encode_image_to_base64(image_path)

                interpretation = self.interpret_result(cleaned_query, result_str, complexity)
                analysis_metadata["llm_answer"] = interpretation

                output_file_path = os.path.join(self.output_folder, "analysis_results.json")
                with open(output_file_path, "w") as output_file:
                    json.dump(analysis_metadata, output_file, indent=2)
                print_and_log(f"Analysis results saved to {output_file_path}")
                return interpretation

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_attempts - 1:
                    print_and_log("Attempting to generate alternative analysis due to failure.")
                    alternative_code = self.generate_alternative_analysis(cleaned_query)
                    alternative_result_str = execute_python_code(extract_code_blocks(alternative_code), self.output_folder)
                    interpretation = self.interpret_result(cleaned_query, alternative_result_str, "ALTERNATIVE_ANALYSIS")
                    analysis_metadata["llm_answer"] = interpretation
                    output_file_path = os.path.join(self.output_folder, "alternative_analysis_results.json")
                    with open(output_file_path, "w") as output_file:
                        json.dump(analysis_metadata, output_file, indent=2)
                    print_and_log(f"Alternative analysis results saved to {output_file_path}")
                    return interpretation

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
        Please provide a comprehensive response to the user's question based on this result. Structure the response as follows:
        1. Start with a complete and direct response to the user's question and explain why this answers the question.
        2. Detail associated with the answers to user's question
        3. Optional: Anything such as limitation and calculation process.
        """
        return call_generative_api(prompt)

    def describe_visualization(self, output_folder: str) -> str:
        """Generates a description of the visualization."""
        visualization_files = [f for f in os.listdir(output_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.svg'))]
        if not visualization_files:
            return "No visualization generated."
        
        description = f"Generated visualizations: {', '.join(visualization_files)}."
        description += " These visualizations represent the data analysis and insights as requested by the user query."
        return description

    def generate_alternative_analysis(self, user_query: str) -> str:
        """Generates alternative analysis or visualization based on the user's query intent."""
        prompt = f"""
        Generate an alternative analysis or visualization based on the user's query intent: '{user_query}'
        Database structure: {json.dumps(self.db_structure, indent=2)}
        Sample Data: {json.dumps(self.sample_data, indent=2)}
        Guidelines:
        1. The .db file is in the current working directory. Use glob to find it.
        2. Use SQLite to query the database and pandas for data manipulation.
        3. Create visualizations using matplotlib or seaborn or anything you see fit.
        4. Handle potential errors such as missing data or unexpected data types.
        4. When dealing with complex tasks in which calculation is required before visualization, always get all the data needed first and then visualize.
        5. Optimize for performance by only reading necessary data from the database.
        6. For calculations like averages, exclude null values and potentially invalid values (e.g., zero for age).
        7. Provide counts of total records, valid records, and records excluded due to null or invalid values.
        8. Include comments explaining any assumptions or decisions made in the analysis.
        9. Save all generated visualizations and data outputs to the '{self.output_folder}' directory.
        10. Ensure the visualization is meticulous with only the right contents and is friendly to the eyes, and academic publication ready.
        11. Pay attention to the font size, color, placement of texts, font size, and ensure the font can display multilingual languages if applied.
        12. Ensure maximum aesthetic, modernity, professionalism, and top-notched academic journal publication-ready quality.
        Please respond with the code inside #begin and #end markers.
        """
        return call_generative_api(prompt)

def main():
    try:
        # Update pip to the latest version
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print_and_log("Updated pip to the latest version.")

        db_files = glob.glob("*.db")
        if not db_files:
            raise FileNotFoundError("No .db files found in the current directory.")
        
        print("Available .db files:")
        for idx, file in enumerate(db_files):
            print(f"{idx + 1}. {file}")
        choice = int(input("Select a .db file by number: ")) - 1
        db_file = db_files[choice]

        user_query = input("Enter your query: ")

        data_tool = DataAnalysisTool(db_file, user_query)
        print(f"Database file: {data_tool.db_file}")
        print(f"Database structure:\n{json.dumps(data_tool.db_structure, indent=2)}")
        
        result = data_tool.analyze(user_query)
        print("\nAnalysis Result:")
        print(result)
        print("\n" + "="*50)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
