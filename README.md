# Chat-with-SQLite-via-LLM

This script leverages the Gemini LLM to interact directly with SQLite files (.db) by generating either an SQLite query or an automated Python script to retrieve results. These results can range from simple calculations to complex visualizations. For example, using a sample SQLite file containing FINRA short volume data for 2024-08-23, the script can answer a query to find and visualize the top ten most shorted stocks. The LLM efficiently utilizes the SQLite dataset and automatically generates Python code to produce the final result—all with a single click.

## Features

- **Direct SQLite Interaction:** Generate SQLite queries or Python scripts to analyze data.
- **Automated Python Code Generation:** Automatically generates Python code to handle queries.
- **Visualization Support:** Generate visualizations for complex data analysis.

## Getting Started

To use this script with any SQLite file, follow these steps:

### 1. Obtain an API Key

Obtain a free API key from Google Gemini:

- [Get API Key](https://ai.google.dev/gemini-api/docs/api-key)

### 2. Configure the Script

Replace the `API_KEY` placeholder in the code with your actual API key. 

### 3. Install Required Packages

Install the necessary Python packages by running:

```sh
pip install pandas numpy google-generativeai
```
### 4. Modify the User Query
To ask a different question about the SQLite file or request different visualizations:

Open the script file.

Locate the line where the hardcoded user_query is defined. It will look something like this:

python
Copy code
# Hardcoded user query
user_query = "find a way to visualize those high short volume stock and put their name on top of it and show all the symbols depending on their short vol size put in different y axis etc"
# Replace the query string with your desired question about the SQLite file or the specific visualization you want. For example:
# Modify the user query as needed
user_query = "Show a trend analysis of stock prices over time with annotations for significant price changes."

### 5. Prepare Your SQLite File
Use the sample FINRA short volume SQLite file for testing or replace it with any SQLite file of your choice. Save the file in the same directory as the script.

API Usage Limits
As of August 2024, Google Gemini offers up to 1.5 billion free API tokens per day, allowing for extensive use of the API.

Example Usage
Run the script with your SQLite file.
Input your query when prompted.
Review the generated output and visualizations.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Google Gemini API for providing the generative language model.
The pandas and numpy libraries for data manipulation.
```
![05359ccf52ac2baffb69e50d90e84db](https://github.com/user-attachments/assets/d13630ca-d890-4835-89c6-0f317da77fca)
![7c2318351594cd5f7b8dd8fd33ebdc9](https://github.com/user-attachments/assets/738680fa-9ce3-4356-a50c-9aa60140b708)


