# Chat-with-SQLite-via-LLM

This script utilizes the Gemini LLM to interact directly with SQLite files (.db). It can generate SQLite queries or automated Python scripts to retrieve and analyze data, ranging from simple calculations to complex visualizations. For example, using a sample SQLite file with FINRA short volume data from 2024-08-23, the script can answer a query to find and visualize the top ten most shorted stocks. The LLM efficiently uses the SQLite dataset and automatically generates Python code to produce the final result—all with a single click.

## Features

- **Direct SQLite Interaction:** Generate SQLite queries or Python scripts to analyze data.
- **Automated Python Code Generation:** Automatically create Python code to handle queries.
- **Visualization Support:** Produce visualizations for in-depth data analysis.

## Getting Started

To use this script with any SQLite file, follow these steps:

### 1. Obtain an API Key

Get a free API key from Google Gemini:

- [Obtain API Key](https://ai.google.dev/gemini-api/docs/api-key)

### 2. Configure the Script

Replace the `API_KEY` placeholder in the script with your actual API key.

### 3. Install Required Packages

Install the necessary Python packages by running:

```sh
pip install pandas numpy google-generativeai

### 4. Modify the User Query
To ask a different question about the SQLite file or request different visualizations:

Open the script file.

Locate the line where user_query is defined. It will look like this:

# Hardcoded user query
user_query = "find a way to visualize those high short volume stock and put their name on top of it and show all the symbols depending on their short vol size put in different y axis etc"
Replace the query string with your desired question or visualization. For example:
# Modify the user query as needed
user_query = "Show a trend analysis of stock prices over time with annotations for significant price changes."
5. Prepare Your SQLite File
Use the provided sample FINRA short volume SQLite file for testing, or replace it with any SQLite file of your choice. Save the file in the same directory as the script.


Example Usage
Run the script with your SQLite file.
Input your query when prompted.
Review the generated output and visualizations.


```
![05359ccf52ac2baffb69e50d90e84db](https://github.com/user-attachments/assets/f701ae95-f787-44aa-9635-4a0ea79e3f7d)

![7c2318351594cd5f7b8dd8fd33ebdc9](https://github.com/user-attachments/assets/209291e0-971d-4ac7-8070-f74200798866)

```
### Acknowledgements

- **Google Gemini API:** For providing the generative language model.
- **pandas and numpy libraries:** For their powerful data manipulation capabilities.

### API Usage Limits

As of August 2024, Google Gemini provides up to 1.5 billion free API tokens per day.

### License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

### Inspiration

This code was inspired by the data analysis agent examples from Hugging Face’s [Aymeric Roucher](https://huggingface.co/spaces/m-ric/agent-data-analyst) and Google’s [Dipanjan S.](https://huggingface.co/learn/cookbook/agent_text_to_sql). Special thanks for sharing their expertise and code.

```
