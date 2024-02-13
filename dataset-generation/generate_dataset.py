import requests
import pandas as pd

url = "https://datasets-server.huggingface.co/rows?dataset=chibbss%2Ffitness-chat-prompt-completion-dataset&config=default&split=train&offset=0&length=100"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    rows_data = data.get("rows", [])

    extracted_data = []

    # Loop through each row and extract the "output" and "instruction" fields
    for row in rows_data:
        row_data = row.get("row", {})
        output = row_data.get("output", "")
        instruction = row_data.get("instruction", "")
        extracted_data.append((output, instruction))

    df = pd.DataFrame(extracted_data, columns=['output', 'instruction'])
    df.to_csv('embedding/dataset.csv', index=False)
    print("Dataset saved as dataset.csv")
else:
    print("Failed to retrieve the dataset. Status code:", response.status_code)


