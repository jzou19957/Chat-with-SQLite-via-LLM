# Chat-with-SQLite-via-LLM
This script leverages the Gemini LLM to interact directly with SQLite files (.db) by generating either an SQLite query or an automated Python script to retrieve results. These results can range from simple calculations to complex visualizations. For example, using a sample SQLite file containing FINRA short volume data for 2024-08-23, the script can answer a query to find and visualize the top ten most shorted stocks. The LLM efficiently utilizes the SQLite dataset and automatically generates Python code to produce the final result—all with a single click.

        To use this script with any SQLite file, follow these steps:
          1. Obtain a free API key from Google Gemini: Get API Key.
          2. Replace the API_KEY placeholder in the code with your actual API key.
          3. Install the required Python packages by running:
              pip install pandas numpy google-generativeai

![05359ccf52ac2baffb69e50d90e84db](https://github.com/user-attachments/assets/a102c9c0-8a0a-48cf-b9ad-25a78a8e33fe)
![7c2318351594cd5f7b8dd8fd33ebdc9](https://github.com/user-attachments/assets/fb19041f-8ad5-42fc-938c-f0d84e0d41b4)
