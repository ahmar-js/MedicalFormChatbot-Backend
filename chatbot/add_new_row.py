from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from django.http import JsonResponse

import os, json, csv
import pandas as pd

from dotenv import find_dotenv, load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(model="gpt-4", temperature=0)
llm = ChatOpenAI(model="gpt-4", temperature=0)

embeddings = OpenAIEmbeddings()
chroma_db_path = "media/db/jupyter-chatbot-large"
temp_df = pd.DataFrame()
# chroma_db_path = "jupyter-chatbot-small"
def remove_quotes(s):
    if isinstance(s, str):
        return s.strip('"')
    return s
def check_row(csv_file_path, new_row):
    fieldnames = ['patient_id', 'name', 'dob', 'age', 'gender', 'weight', 'health_conditions', 'address_street', 'address_city', 'phone_number', 'emergency_contact_name', 'emergency_contact_relation', 'emergency_contact_phone_number', 'patient_complaint', 'medical_history', 'medications', 'allergies',
                  'Lifestyle_Information', 'Insurance_Information', 'Vital_Signs', 'Form_Name']

    try:
        existing_df = pd.read_csv(csv_file_path, keep_default_na=False)
        # temp_df = existing_df.copy()
    except FileNotFoundError:
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        existing_df = pd.DataFrame(columns=fieldnames)
        # temp_df = existing_df.copy()



    new_row_df = pd.DataFrame([new_row])
    # Remove leading/trailing spaces
    existing_df = existing_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    new_row_df = new_row_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Ensure consistent data types
    new_row_df = new_row_df.astype(str)
    existing_df = existing_df.astype(str)
    temp_df = existing_df.copy()
    print(temp_df)
    if not existing_df.empty:
        print("here")
        print(existing_df)
        new_row_df = new_row_df.applymap(remove_quotes)
        before_count = len(existing_df)
        print("Before count:", before_count)
        # .drop_duplicates()
        updated_df = pd.concat([temp_df, new_row_df], ignore_index=True)
        updated_df = updated_df.drop(columns=['Form_Name'], errors='ignore')
        updated_df.drop_duplicates(inplace=True)
        after_count = len(updated_df)
        print("after_count:", after_count)
        if after_count > before_count:
            print(temp_df['allergies'])

            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            print(new_row_df.to_dict())
            # updated_df_temp = updated_df_temp.drop(columns=['Form_Name'], errors='ignore')
            updated_df.to_csv(csv_file_path, index=False)
            print("No Duplicate")
            return True
        else:
            print("Duplicate")
            return False
    else:
        new_row_df = new_row_df.applymap(remove_quotes)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        # updated_df_temp = updated_df_temp.drop(columns=['Form_Name'], errors='ignore')
        updated_df.to_csv(csv_file_path, index=False)
        return True

            # Save the updated DataFrame back to the CSV file
            # updated_df.to_csv(csv_file_path, index=False)


    # def row_exists(existing_df, new_row):
    #     new_row = pd.DataFrame([new_row])
    #     new_row = new_row.applymap(remove_quotes)
    #     existing_df_filtered = existing_df.drop(columns=['Form_Name'], errors='ignore')
    #     new_row_filtered = new_row.drop(columns=['Form_Name'], errors='ignore')
    #     print(new_row_filtered.iloc[0])
    #     return existing_df_filtered.apply(lambda row: row.equals(new_row_filtered.iloc[0]), axis=1).any()

    # def row_exists(existing_df, new_row):
    #     print(type(existing_df), type(new_row))
    #     new_row = pd.DataFrame([new_row])

    #     existing_df = existing_df.drop('Form_Name', axis=1)
    #     new_row_temp = new_row.drop('Form_Name', axis=1)
    #     # return existing_df.apply(lambda row: row.to_dict() == new_row_temp.to_dict(), axis=1).any()
    #     return existing_df.apply(lambda row: row.equals(new_row_temp.iloc[0]), axis=1).any()


    # if not row_exists(existing_df, new_row):
    #     # If the new row does not exist, append it to the DataFrame
    #     updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    #     # Save the updated DataFrame back to the CSV file
    #     updated_df.to_csv(csv_file_path, index=False)

    #     print("New row appended to the CSV file.")
    #     return True
    # else:
    #     print("The row already exists in the CSV file.")
    #     return False
    
def add_document(documents):
    vectorstore = Chroma(
        # collection_name="my_collection_small",
        collection_name="my_collection_large",
        embedding_function=embeddings,
        persist_directory=chroma_db_path,
    )

    id = vectorstore._collection.count() + 1
    print(vectorstore._collection.count())
    texts = [doc.page_content for doc in documents]
    vectorstore.add_texts(texts=texts, ids=[str(id)])
    print(vectorstore._collection)
    print(vectorstore._collection.count())

def add_doc(csv_file_path, new_row):
    # new_row["name"] = "William Roger"
    if check_row(csv_file_path, new_row):

        data_file = CSVLoader(csv_file_path)  # Use CSV Loader for CSV files
        docs = data_file.load()
        new_docs = [docs[-1]]

        add_document(new_docs)
        return "Record Added Successfully"
    else:
        return "Record Already Exist"


# check_row('jupyter-ahmar.csv', {'hello': 'H'})
# json_string = json.loads("new_record.json")
# print(add_doc('jupyter-ahmar.csv', {'hello': 'H'}))

# with open("new_record.json", 'r') as file:
#     data = json.load(file)

# print(add_doc('jupyter-ahmar-large.csv', data))
# check_row('jupyter-ahmar.csv', dict())