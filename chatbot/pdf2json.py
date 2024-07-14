import os
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_extraction_chain, create_extraction_chain_pydantic
import json
from typing import List

load_dotenv()

llm = ChatOpenAI()

class Document(BaseModel):
    patient_id: str = Field(description="patient id")
    name: str = Field(description="name")
    dob: str = Field(description="date of birth")
    age: str = Field(description="age")
    gender: str = Field(description="gender")
    weight: str = Field(description="weight")
    health_conditions: str = Field(description="health conditions")
    address_street: str = Field(description="address street")
    address_city: str = Field(description="address city")
    phone_number: str = Field(description="phone number")
    emergency_contact_name: str = Field(description="emergency contact name")
    emergency_contact_relation: str = Field(description="emergency contact relation")
    emergency_contact_phone_number: str = Field(description="emergency contact phone number")
    patient_complaint: str = Field(description="Patient Complaint")
    medical_history: List[str] = Field(description="medical history")
    medications: List[str] = Field(description="medications")
    allergies: List[str] = Field(description="allergies")
    Lifestyle_Information: List[str] = Field(description="Lifestyle Information")
    Insurance_Information: List[str] = Field(description="Insurance Information")
    Vital_Signs: List[str] = Field(description="Vital Signs")
    # file_path: str = Field(description="path of the source PDF file")

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

def remove_pdf_extension(file_name):
    if file_name.lower().endswith('.pdf'):
        return file_name[:-4]  # Remove the last 4 characters (i.e., '.pdf')
    return file_name

def extract_metadata_from_pdf_output_parser(file_path):
    pages = load_pdf(file_path)
    
    parser = JsonOutputParser(pydantic_object=Document)
    
    prompt = PromptTemplate(
        template="Extract the information as specified.\n{format_instructions}\n{context}\n",
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    response = chain.invoke({
        "context": pages
    })
    # response_data = {}
    # Add file path to the response data
    response_data = response
    print("file path: ", file_path)
    file_path = remove_pdf_extension(file_path)
    # print(response_data)
    response_data['Form_Name'] = file_path

    
    # response_data.append(file_path)

    class CustomEncoder(json.JSONEncoder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def default(self, o):
            if isinstance(o, str):
                try:
                    return o.encode('utf-8', 'strict').decode('utf-8')
                except UnicodeEncodeError:
                    return repr(o)[1:-1] 
            return super().default(o)
        
    print("Function: extract_metadata_from_pdf_output_parser")
    output_file = os.path.join('media/json', f'{os.path.splitext(os.path.basename(file_path))[0]}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        # json.dump(response, f, ensure_ascii=False, indent=4)
        json.dump(response, f, ensure_ascii=False, indent=4, cls=CustomEncoder)
    return output_file
    # print(response)


    # output_file = os.path.join('media', f'{os.path.splitext(os.path.basename(file_path))[0]}.json')
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(response_data, f, ensure_ascii=False, indent=4, cls=CustomEncoder)
    # print(f"Extracted data from {file_path} to {output_file}")
    
def process_all_pdfs_in_folder(folder_path):
    json_path = None
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            # print(file_path)
            json_path = extract_metadata_from_pdf_output_parser(file_path)
            # # Delete pdf after converting to json
            # os.remove(file_path)
    return json_path

def process_single_pdf(file_path):
    return extract_metadata_from_pdf_output_parser(file_path)

# pdf_folder = "media/"

# process_all_pdfs_in_folder(pdf_folder)
