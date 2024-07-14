import re
# from django.shortcuts import render
from django.http import JsonResponse
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI 
from langchain_community.embeddings import OpenAIEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
# from langchain_community.document_loaders import CSVLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
import pandas as pd
from .add_new_row import add_doc
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
# from rest_framework.response import Response
# from rest_framework import status
from django.http import FileResponse
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views import View
# from rest_framework.views import APIView
from .pdf2json import process_all_pdfs_in_folder, process_single_pdf
from .add_new_row import add_doc
from openai import OpenAI
# import uuid
client = OpenAI()
import os, json
from dotenv import load_dotenv
import boto3
from django.conf import settings
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0)

# file_path = "media/csv/jupyter-ahmar-large.csv"
# data_file = CSVLoader(file_path)
# docs = data_file.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# splits = text_splitter.split_documents(docs)

chroma_db_path = "media/db/jupyter-chatbot-large"
embeddings = OpenAIEmbeddings()
session_id = "abc1234"

@method_decorator(csrf_exempt, name='dispatch')
class FileUploadView(View):
    def post(self, request, *args, **kwargs):
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        file_name = file.name
        print(file_name)
        file_name = default_storage.save(file.name, ContentFile(file.read()))
        file_url = default_storage.url(file_name)
        print("file name: ", file_name)
        print("file_url: ", file_url)
        print("PDF Uploaded")
        json_path = process_single_pdf('media/' + file_name)
        print("Converted to json")
        with open(json_path, 'r') as jsonfile:
            data = json.load(jsonfile)
            data['medical_history'] = "; ".join(data['medical_history'])
            data['medications'] = "; ".join(data['medications'])
            data['allergies'] = "; ".join(data['allergies'])
            data['Lifestyle_Information'] = "; ".join(data['Lifestyle_Information'])
            data['Insurance_Information'] = "; ".join(data['Insurance_Information'])
            data['Vital_Signs'] = "; ".join(data['Vital_Signs'])
        # data = pd.DataFrame([data])
        add_doc('media/csv/jupyter-ahmar-large.csv', data)
        # print(adddoc)
        print("Record added in csv")
        bucket_name = settings.AWS_STORAGE_BUCKET_NAME
        s3_file_key = f'{file_name}'  # Uploading to media/ directory in S3
        print("bucket name: ", bucket_name)
        print("file_key: ", s3_file_key)
        file_path = f'media/{file_name}'
        print("file path", file_path)
        # Call the utility function to upload the file to S3
        upload = upload_file_to_s3(file_path, bucket_name, s3_file_key)
        if upload:
            print("Success")
        else:
            print("error")
        # os.remove(json_path)
        # print("Json removed")
        return JsonResponse({'file_name': file_name, 'file_url': file_url})



def add_document(documents):
    vectorstore = Chroma(
        collection_name="my_collection_small",
        embedding_function=embeddings,
        persist_directory=chroma_db_path,
    )
    texts = [doc.page_content for doc in documents]
    ids = [str(i) for i, doc in enumerate(documents)]
    vectorstore.add_texts(texts=texts, ids=ids)
    # print("count", vectorstore._collection.count())
def load_collection():
    loaded_vectorstore = Chroma(
        collection_name="my_collection_large",
        embedding_function=embeddings,
        persist_directory=chroma_db_path,
    )
    # print("Loaded vector: ", loaded_vectorstore)
    return loaded_vectorstore

def vectorstore_retriever(vectorstore):
    print("vector store retriever start")
    # print("Ab main function kay start main hu")
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # print("vectorstore retriever: ", vectorstore_retriever )
    # print("Ab main function kay thora sa agay hu")

    docs = [Document(i) for i in vectorstore.get()['documents']]
    # print("abracadabra: ", vectorstore.get())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)
    # print("Thora or agay")
    # print(splits)
    keyword_retriever = BM25Retriever.from_documents(splits)
    # print("Ab main function kay mid main hu")

    keyword_retriever.k = 3
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )
    print("vector store retriever end")
    # print("Ab main function kay end main hu")

    return ensemble_retriever

def contextualize_question(ensemble_retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, ensemble_retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def documents_chain(history_aware_retriever):
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. "
        "Do not answer questions out of context"
        "If you don't know the answer, say that you don't know. "
        "NOTE: there can be multiple patients with the same id or name, return all patients that match the query and ask the user which one they want."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def save_session_history(session_id, history):
    # Ensure we only keep the last 5 conversations
    if len(history) > 10:
        history = history[-10:]
    
    with open(f"{session_id}_history.json", "w") as f:
        json.dump(history, f)

def load_session_history(session_id):
    try:
        with open(f"{session_id}_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def get_session_history(session_id):
    return load_session_history(session_id)

def save_conversation_history(session_id, conversation_history):
    history = get_session_history(session_id)
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        history.append({
            "role": role,
            "content": message.content
        })
    save_session_history(session_id, history)

def conversational_chain(rag_chain, session_id):
    chat_history = get_session_history(session_id)
    chat_message_history = ChatMessageHistory()
    for message in chat_history:
        chat_message_history.add_message(message["role"])
        chat_message_history.add_message(message["content"])
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda _: chat_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain, chat_message_history

def handle_user_input(rag_chain, session_id, user_input):
    print("Handle user input  start")
    conversation_chain, chat_message_history = conversational_chain(rag_chain, session_id)
    response = conversation_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": session_id}
        }
    )
    chat_message_history.add_message(user_input)
    chat_message_history.add_message(response["answer"])
    save_conversation_history(session_id, chat_message_history.messages)
    # print("Ye rha handle user input kay end main")
    # print("Response: ", response['answer'])
    # print(f"Type of response: {type(response)}")
    # print("chatbot answer: ", chatbot_answer)
    # print("complete response: ", response)
    chatbot_answer = response['answer']
    # print("huhu")
    chatbot_answer = chatbot_answer.replace('"', '')
    chatbot_answer = chatbot_answer.replace('.', '')
    # chatbot_answer = chatbot_answer.replace('*', '')
    # chatbot_answer = chatbot_answer.replace('**', '')
    # return response['answer']
    # print("chatbot answer from handle user input", chatbot_answer)
    print("Handle user input  end")
    return chatbot_answer

def answer_validation(rag_chain, session_id, chatbot_answer):
    print("Answer validation start")
    query = f"Here is the query: {chatbot_answer}"

    gpt_response = client.chat.completions.create(
      model="gpt-4-turbo",
      messages=[
        {"role": "system", "content": "You are given a query, if the query mentions it did not find a record for an ID return the ID otherwise return 0."},
        {"role": "user", "content": f"{query}"}
        ]
    )
    # print("query response: ", query)
    gpt_response = gpt_response.choices[0].message.content.strip()
    # print("I am a gpt response after query response: ", gpt_response)
    # print("Type of gpt response: ", type(gpt_response))
    # print(response.choices[0].message.content)

    if gpt_response != '0':
        csv_file = pd.read_csv("media/csv/jupyter-ahmar-large.csv")
        matching_rows = csv_file[csv_file['patient_id'] == gpt_response]
        # print("Matching rows name values: ", matching_rows['name'].values)
        fetched_names = matching_rows['name'].values

        new_query = f"You said there are no records for id {gpt_response}, but these names have this id {fetched_names}"

        response = handle_user_input(rag_chain, session_id, f"Note: there can be multiple patients with the same id or name, return all patients that match the query and ask the user which one they want.\n\nAlways return the form name for the patient in single asterik.\n\nUser Query: {new_query}.")
        # print(response["answer"])
        print("Answer validation end")
        
        print
        return response
    else:
        # print("Response from answer validation else block", chatbot_answer)
        print("Answer validation end")

        return chatbot_answer
            # print("gpt response inside answer validation:", gpt_response)
            # return response['answer']

# print("I am ahmer")
# add_document(splits)
# def extract_form_name(response_text):
#     match = re.search(r"Form_Name:\s(.+?)(\n|$)", response_text)
#     return match.group(1) if match else None

def extract_form_name(response):
    pattern = r"\*(.*?)\*"
    if len(pattern):
        if not len(pattern[0]):
            pattern = r"\**(.**?)\**"
    # Use re.findall to find all matches
    matches = re.findall(pattern, response)
    # print(matches)
    def trim_to_p(s):
        # Find the index of the first occurrence of 'p' or 'P'
        index_p = s.lower().find('p')
        # If 'p' or 'P' is not found, return an empty string
        if index_p == -1:
            return ""
        # Return the substring starting from the first 'p'
        return s[index_p:]
    pdf_names = [trim_to_p(i) for i in matches]
    lenght_of_pdfs = len(pdf_names)
    if lenght_of_pdfs == 1:
        return pdf_names[0]
    else:
        return "No Pdfs found"


def upload_file_to_s3(file_path, bucket_name, s3_file_key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_S3_REGION_NAME,
    )

    try:
        s3.upload_file(file_path, bucket_name, s3_file_key, ExtraArgs={
                'ContentType': 'application/pdf',
                'ContentDisposition': 'inline'
            })
        print(f"Upload Successful: {s3_file_key}")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except PartialCredentialsError:
        print("Incomplete credentials provided")
        return False

def chat(request):
    try:
        # print("ahmer")
        loaded_vectorstore = load_collection()
        # print("Loaded collections: ", loaded_vectorstore)
        # print("main yaha hoo")
        # print(loaded_vectorstore.get())
        ensemble_retriever = vectorstore_retriever(loaded_vectorstore)
        # print(ensemble_retriever)
        # print("Ab main yaha hoo")

        history_aware_retriever = contextualize_question(ensemble_retriever)
        rag_chain = documents_chain(history_aware_retriever)
        user_input = request.GET.get('query', '')
        session_id = request.session.session_key or "abc1234"
        
        # print("here")
        response = handle_user_input(rag_chain, session_id, f"Note: there can be multiple patients with the same id or name, return all patients that match the query and ask the user which one they want.\n\nAlways return the form name for the patient in single asterik.\n\nUser Query: {user_input}.")
        # print("here2")
        # print(response)
        answerValidation = answer_validation(rag_chain, session_id, response)
        form_name = extract_form_name(answerValidation)
        # print(f"media/{form_name}.pdf")
        # pdf_name = f"{form_name}.pdf"
        
        file_path = f"https://patientintakeforms.s3.amazonaws.com/{form_name}.pdf"
        # file_path = f"media/{form_name}.pdf"
        
        # pdf_path = f'{request.build_absolute_uri("/media/")}{form_name}.pdf'
        # print(pdf_path)
#                     response = FileResponse(pdf, content_type='application/pdf')
#                     response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
#                     return response
        # Generate a unique token
        # download_token = str(uuid.uuid4())
        # print("Session ID:", request.session.session_key)
        # request.session['download_token'] = download_token
        # request.session['file_name'] = pdf_path
        # print("download_token: ", download_token)
        # request.session.save()
        # print("Session data after setting:", request.session.items())
        # print(f"media/"form_name)
        # print("AnswerValidation: ", answerValidation)
        # return JsonResponse({"response": answerValidation, "download_token": download_token})
        # bucket_name = settings.AWS_STORAGE_BUCKET_NAME
        # s3_file_key = f'media/{pdf_name}'  # Uploading to media/ directory in S3
        
        # # Call the utility function to upload the file to S3
        # upload = upload_file_to_s3(file_path, bucket_name, s3_file_key)
        # if upload:
        #     print("Success")
        # else:
        #     print("error")
        return JsonResponse({"response": answerValidation, "pdf_path": file_path})
        
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# from rest_framework.permissions import AllowAny
# class DownloadPDF(APIView):
#     # permission_classes = [AllowAny]
#     def get(self, request, *args, **kwargs):
#         # download_token = request.GET.get('token', '')
#         # session_token = request.session.get('download_token', '')
#         # pdf_path = request.session.get('file_name', '')
#         # print("pdf_path when downloading: ", pdf_path)
#         # print("session token", session_token)
#         # print("Download Token:", download_token)
#         # print("Session ID:", request.session.session_key)

#         # print("Session data when downloading:", request.session.items())

#         # if not pdf_path:
#         #     return Response({"error": "Path not found"}, status=status.HTTP_404_NOT_FOUND)
#         if download_token == session_token:
#             file_path = pdf_path  # Update this to dynamically use the form_name
#             if os.path.exists(file_path):
#                 with open(file_path, 'rb') as pdf:
#                     response = FileResponse(pdf, content_type='application/pdf')
#                     response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
#                     return response
#             return Response({"error": "File not found"}, status=status.HTTP_404_NOT_FOUND)
        
#         return Response({"error": "Invalid or missing token"}, status=status.HTTP_403_FORBIDDEN)