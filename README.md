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
![05359ccf52ac2baffb69e50d90e84db](https://github.com/user-attachments/assets/d13630ca-d890-4835-89c6-0f317da77fca)
![7c2318351594cd5f7b8dd8fd33ebdc9](https://github.com/user-attachments/assets/738680fa-9ce3-4356-a50c-9aa60140b708)


