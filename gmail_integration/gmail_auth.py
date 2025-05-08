import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    # Dynamically locate credentials.json from project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CREDENTIALS_PATH = os.path.join(BASE_DIR, 'secrets', 'credentials.json')

    flow = InstalledAppFlow.from_client_secrets_file(
        CREDENTIALS_PATH,
        SCOPES
    )
    creds = flow.run_local_server(port=0)
    service = build('gmail', 'v1', credentials=creds)
    return service