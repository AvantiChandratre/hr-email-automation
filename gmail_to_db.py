import os
import imaplib
import email
from email.header import decode_header
import re
import psycopg2
from datetime import datetime
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from transformers import pipeline
import urllib3
import certifi
import hashlib
import json
import requests
from langchain.prompts import PromptTemplate
#from langchain_community.llms import HuggingFaceEndpoint
from langchain import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Constants for classification
ZS_THRESHOLD = 0.2
SIMILARITY_THRESHOLD = 0.75

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()

# Initialize sentence transformer model
encoder = SentenceTransformer('all-mpnet-base-v2')

def connect_to_postgres():
    """Connect to Google Cloud PostgreSQL instance."""
    try:
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST'),
            port=os.environ.get('DB_PORT', 5432),
            database=os.environ.get('DB_NAME'),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASSWORD')
        )
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def verify_existing_table(conn):
    """Verify the existing email table has all required columns."""
    with conn.cursor() as cursor:
        # Check if the table exists
        cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'emails'
        );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("Table 'emails' does not exist. Creating table...")
            cursor.execute("""
            CREATE TABLE emails (
                email_id SERIAL PRIMARY KEY,
                sender_email VARCHAR(255),
                subject TEXT,
                body TEXT,
                received_at TIMESTAMP,
                classified_category VARCHAR(50),
                status VARCHAR(20),
                escalated BOOLEAN,
                thread_id BIGINT,
                classification_confidence FLOAT,
                response_email TEXT,
                learn BOOLEAN,
                similarity_source_email INTEGER,
                similarity_to_learned FLOAT,
                message_id VARCHAR(255),
                responded_at TIMESTAMP
            );
            """)
            
            # Create learned_email_vectors table if it doesn't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_email_vectors (
                email_id INTEGER PRIMARY KEY,
                category VARCHAR(50),
                vector FLOAT[]
            );
            """)
            
            conn.commit()
            print("Tables created successfully.")
            return True
            
        # Check if all required columns exist
        cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'emails';
        """)
        
        existing_columns = {row[0] for row in cursor.fetchall()}
        required_columns = {
            'email_id', 'sender_email', 'subject', 'body', 'received_at',
            'classified_category', 'status', 'escalated', 'thread_id',
            'classification_confidence', 'response_email', 'learn',
            'similarity_source_email', 'similarity_to_learned',
            'message_id', 'responded_at'
        }
        
        missing_columns = required_columns - existing_columns
        
        if missing_columns:
            print(f"Adding missing columns: {missing_columns}")
            for column in missing_columns:
                if column == 'email_id':
                    continue  # Skip primary key
                
                # Define column type based on name
                if column in {'sender_email', 'status', 'classified_category', 'message_id'}:
                    col_type = 'VARCHAR(255)'
                elif column in {'subject', 'body', 'response_email'}:
                    col_type = 'TEXT'
                elif column in {'received_at', 'responded_at'}:
                    col_type = 'TIMESTAMP'
                elif column in {'escalated', 'learn'}:
                    col_type = 'BOOLEAN'
                elif column in {'thread_id', 'similarity_source_email'}:
                    col_type = 'INTEGER'
                elif column in {'classification_confidence', 'similarity_to_learned'}:
                    col_type = 'FLOAT'
                else:
                    col_type = 'TEXT'  # Default type
                
                cursor.execute(f"""
                ALTER TABLE emails 
                ADD COLUMN IF NOT EXISTS {column} {col_type};
                """)
            
            conn.commit()
            print("Missing columns added successfully.")
        
        # Check if learned_email_vectors table exists
        cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'learned_email_vectors'
        );
        """)
        
        learned_vectors_exists = cursor.fetchone()[0]
        
        if not learned_vectors_exists:
            print("Creating learned_email_vectors table...")
            cursor.execute("""
            CREATE TABLE learned_email_vectors (
                email_id INTEGER PRIMARY KEY,
                category VARCHAR(50),
                vector FLOAT[]
            );
            """)
            conn.commit()
            print("Learned vectors table created successfully.")
            
        return True

def clean_text(text):
    """Clean and normalize text content."""
    if text is None:
        return ""
    
    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])
    
    return text.strip()

def decode_email_header_text(header):
    """Decode email header."""
    if header is None:
        return ""
    
    result = ""
    decoded_header = decode_header(header)
    
    for content, encoding in decoded_header:
        if isinstance(content, bytes):
            try:
                if encoding:
                    content = content.decode(encoding)
                else:
                    content = content.decode('utf-8', errors='replace')
            except Exception:
                content = content.decode('utf-8', errors='replace')
        result += str(content)
    
    return result

def get_email_body(msg):
    """Extract email body from message."""
    body = ""
    
    if msg.is_multipart():
        # If the email has multiple parts, try to find the text/plain part
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
                
            if content_type == "text/plain":
                try:
                    body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    break
                except Exception as e:
                    print(f"Error decoding plain text: {e}")
            
            # If no text/plain is found, try text/html
            elif content_type == "text/html" and not body:
                try:
                    html_body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    # Simple HTML tag removal - consider using BeautifulSoup for better results
                    body = re.sub(r'<[^>]+>', ' ', html_body)
                except Exception as e:
                    print(f"Error decoding HTML: {e}")
    else:
        # If the email is not multipart
        try:
            body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Error decoding email body: {e}")
            body = ""
    
    return clean_text(body)

def extract_sender_email(from_header):
    """Extract email address from From header."""
    if not from_header:
        return ""
        
    # Try to extract email from format: "Name <email@example.com>"
    email_match = re.search(r'<([^>]+)>', from_header)
    if email_match:
        return email_match.group(1)
    
    # If no angle brackets, return as is (likely just an email address)
    return from_header.strip()

def parse_date(date_str):
    """Parse date from email header."""
    if not date_str:
        return datetime.now()
        
    try:
        # Try to parse various date formats
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S',
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        # If all formats fail, use current time
        return datetime.now()
    except Exception:
        return datetime.now()

def get_classifier():
    """Initialize and return the email classifier."""
    try:
        # Setup classifier with online model
        classifier = pipeline(
            "zero-shot-classification",
            "facebook/bart-large-mnli"
        )
        return classifier
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def classify_email(classifier, subject, body):
    """Classify an email using the classifier."""
    if not classifier:
        return None, 0.0

    # Define labels
    label_map = {
        "Leave Request": "Requests related to taking leave or vacation, including sick leave, annual leave, or other time off",
        "Onboarding": "Questions about new hire onboarding, joining formalities, or initial setup processes",
        "Job Offer": "Inquiries regarding job offers, employment contracts, or offer letter details",
        "Payroll Inquiry": "Questions related to salary, payslips, payroll processing, or compensation",
        "Benefits Inquiry": "Questions about insurance, medical benefits, or employee benefits programs",
        "Resignation & Exit": "Emails about resignation, exit process, or final settlements",
        "Attendance & Timesheet": "Issues about work hours, attendance tracking, or timesheet submissions",
        "Recruitment Process": "Questions about interview process, screening stages, or hiring procedures",
        "Policy Clarification": "Clarification about company policies, procedures, or guidelines",
        "Training & Development": "Queries about training programs, skill development, or learning opportunities",
        "Work From Home Requests": "Requests or updates regarding remote work arrangements",
        "Relocation & Transfer": "Inquiries about internal transfers, relocation benefits, or location changes",
        "Expense Reimbursement": "Questions about reimbursements, expense claims, or payment processing",
        "IT & Access Issues": "Issues about system access, accounts, or technical problems",
        "Events & Celebrations": "Emails about office events, parties, or company celebrations"
    }

    descriptive_labels = list(label_map.values())
    email_text = f"Subject: {subject}\nBody: {body}"

    try:
        result = classifier(email_text, descriptive_labels)
        predicted_description = result["labels"][0]
        confidence = result["scores"][0]

        # Find the corresponding category
        predicted_label = next(
            key for key, value in label_map.items() if value == predicted_description
        )

        return predicted_label, confidence
    except Exception as e:
        print(f"Error classifying email: {e}")
        return None, 0.0

def get_thread_history(conn, thread_id):
    """Fetch all emails in a thread for context."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT email_id, sender_email, subject, body, received_at, classified_category
        FROM emails
        WHERE thread_id = %s
        ORDER BY received_at ASC
    """, (thread_id,))
    return cursor.fetchall()

def get_category_specific_template(category):
    """Get category-specific template and instructions."""
    templates = {
        "Leave Request": {
            "template": """You are an experienced HR professional handling leave requests. Your role is to provide clear, professional, and empathetic responses while ensuring all necessary information is collected.

CONTEXT:
- You are responding to an employee's leave request
- Your goal is to gather complete information and guide them through the process
- Maintain a supportive and understanding tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: Leave Request
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Opening:
   - Acknowledge the leave request professionally
   - Express understanding of their need for leave

2. Information Gathering:
   - Request specific details if not provided:
     * Start and end dates
     * Type of leave (annual, sick, etc.)
     * Reason for leave
     * Any supporting documentation needed

3. Process Information:
   - Explain the leave approval process
   - Mention expected timeline for approval
   - List required documentation
   - Clarify any policy requirements

4. Next Steps:
   - Provide clear action items for the employee
   - Include submission deadlines if applicable
   - Mention who to contact for questions

5. Closing:
   - Offer additional support
   - Provide contact information for follow-up

Please structure your response in a clear, professional format with appropriate paragraphs and bullet points where needed.""",

            "instructions": "Focus on gathering complete leave details while maintaining a supportive tone. Ensure all necessary information is requested and the approval process is clearly explained."
        },

        "Job Offer": {
            "template": """You are an experienced HR professional handling job offer inquiries. Your role is to provide clear, professional, and accurate information about employment offers and contracts.

CONTEXT:
- You are responding to a job offer inquiry
- Your goal is to provide clear information about the offer and next steps
- Maintain a professional and positive tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: Job Offer
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their interest in the position
   - Confirm understanding of their inquiry

2. Offer Details:
   - Provide clear information about the offer
   - Mention key terms and conditions
   - Include compensation details
   - Specify any benefits included

3. Process Information:
   - Explain the acceptance process
   - Mention any deadlines
   - List required documentation
   - Clarify any conditions

4. Next Steps:
   - Outline the acceptance procedure
   - Include any required actions
   - Mention any forms to complete
   - Specify submission process

5. Support Information:
   - Provide contact information
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points.""",

            "instructions": "Focus on providing clear information about the job offer while maintaining a professional tone. Ensure all necessary details about the offer and acceptance process are clearly communicated."
        },

        "Resignation & Exit": {
            "template": """You are an experienced HR professional handling resignation and exit processes. Your role is to guide employees through the exit process professionally and efficiently.

CONTEXT:
- You are responding to a resignation or exit-related query
- Your goal is to provide clear information about the exit process
- Maintain a professional and supportive tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: Resignation & Exit
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their resignation/exit query professionally
   - Express understanding of their decision

2. Process Information:
   - Explain the exit process steps
   - Mention required documentation
   - Include timeline expectations
   - Clarify any policy requirements

3. Final Settlement:
   - Explain the final settlement process
   - Mention any pending dues
   - Include asset return procedures
   - Specify any clearance requirements

4. Next Steps:
   - Provide clear action items
   - Include submission deadlines
   - Mention required approvals
   - Specify handover procedures

5. Support Information:
   - Provide contact information
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points.""",

            "instructions": "Focus on providing clear information about the exit process while maintaining a professional tone. Ensure all necessary steps and requirements are clearly communicated."
        },

        "Attendance & Timesheet": {
            "template": """You are an experienced HR professional handling attendance and timesheet issues. Your role is to provide clear guidance on attendance policies and timesheet procedures.

CONTEXT:
- You are responding to an attendance or timesheet-related query
- Your goal is to provide clear information about policies and procedures
- Maintain a professional and helpful tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: Attendance & Timesheet
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their attendance/timesheet concern
   - Confirm understanding of their query

2. Policy Information:
   - Explain relevant attendance policies
   - Clarify timesheet requirements
   - Mention any exceptions
   - Include any special cases

3. Process Information:
   - Explain the correction process
   - Mention submission deadlines
   - List required documentation
   - Clarify approval procedures

4. Next Steps:
   - Provide clear action items
   - Include any required forms
   - Mention approval process
   - Specify submission procedures

5. Support Information:
   - Provide contact information
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points.""",

            "instructions": "Focus on providing clear information about attendance policies and timesheet procedures while maintaining a helpful tone. Ensure all necessary steps and requirements are clearly communicated."
        },

        "Recruitment Process": {
            "template": """You are an experienced HR professional handling recruitment process queries. Your role is to provide clear information about the hiring process and candidate status.

CONTEXT:
- You are responding to a recruitment-related query
- Your goal is to provide clear information about the process
- Maintain a professional and informative tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: Recruitment Process
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their recruitment query
   - Confirm understanding of their interest

2. Process Information:
   - Explain the recruitment process
   - Mention current stage
   - Include timeline expectations
   - Clarify any requirements

3. Next Steps:
   - Outline upcoming steps
   - Mention any required actions
   - Include any documentation needed
   - Specify any assessments

4. Status Update:
   - Provide current status
   - Mention any pending items
   - Include any feedback
   - Specify any decisions

5. Support Information:
   - Provide contact information
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points.""",

            "instructions": "Focus on providing clear information about the recruitment process while maintaining a professional tone. Ensure all necessary steps and requirements are clearly communicated."
        },

        "Policy Clarification": {
            "template": """You are an experienced HR professional handling policy clarification requests. Your role is to provide clear and accurate information about company policies.

CONTEXT:
- You are responding to a policy clarification request
- Your goal is to provide clear and accurate policy information
- Maintain a professional and informative tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: Policy Clarification
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their policy question
   - Confirm understanding of their query

2. Policy Information:
   - Provide relevant policy details
   - Explain any exceptions
   - Include any special cases
   - Clarify any ambiguities

3. Application:
   - Explain how the policy applies
   - Mention any conditions
   - Include any requirements
   - Specify any limitations

4. Additional Information:
   - Provide related policies
   - Mention any updates
   - Include any resources
   - Specify any documentation

5. Support Information:
   - Provide contact information
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points.""",

            "instructions": "Focus on providing clear and accurate policy information while maintaining a professional tone. Ensure all necessary details and requirements are clearly communicated."
        },

        "Training & Development": {
            "template": """You are an experienced HR professional handling training and development queries. Your role is to provide information about available programs and development opportunities.

CONTEXT:
- You are responding to a training or development query
- Your goal is to provide clear information about available programs
- Maintain a supportive and encouraging tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: Training & Development
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their interest in development
   - Confirm understanding of their goals

2. Program Information:
   - Provide available program details
   - Explain eligibility criteria
   - Include program schedules
   - Mention any prerequisites

3. Process Information:
   - Explain the enrollment process
   - Mention any deadlines
   - List required documentation
   - Clarify any requirements

4. Next Steps:
   - Provide clear action items
   - Include any required forms
   - Mention approval process
   - Specify submission procedures

5. Support Information:
   - Provide contact information
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points.""",

            "instructions": "Focus on providing clear information about training and development opportunities while maintaining a supportive tone. Ensure all necessary steps and requirements are clearly communicated."
        },

        "Work From Home Requests": {
            "template": """You are an experienced HR professional handling work from home requests. Your role is to provide clear guidance on remote work policies and procedures.

CONTEXT:
- You are responding to a work from home request
- Your goal is to provide clear information about the process
- Maintain a professional and supportive tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: Work From Home Requests
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their WFH request
   - Confirm understanding of their needs

2. Policy Information:
   - Explain WFH policies
   - Mention eligibility criteria
   - Include any restrictions
   - Clarify any requirements

3. Process Information:
   - Explain the approval process
   - Mention any deadlines
   - List required documentation
   - Clarify any conditions

4. Next Steps:
   - Provide clear action items
   - Include any required forms
   - Mention approval process
   - Specify submission procedures

5. Support Information:
   - Provide contact information
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points.""",

            "instructions": "Focus on providing clear information about work from home policies and procedures while maintaining a supportive tone. Ensure all necessary steps and requirements are clearly communicated."
        },

        "Relocation & Transfer": {
            "template": """You are an experienced HR professional handling relocation and transfer requests. Your role is to provide clear information about the transfer process and relocation benefits.

CONTEXT:
- You are responding to a relocation or transfer query
- Your goal is to provide clear information about the process
- Maintain a professional and supportive tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: Relocation & Transfer
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their transfer/relocation request
   - Confirm understanding of their needs

2. Process Information:
   - Explain the transfer process
   - Mention relocation benefits
   - Include timeline expectations
   - Clarify any requirements

3. Documentation:
   - List required documentation
   - Mention any forms needed
   - Include any approvals required
   - Specify any conditions

4. Next Steps:
   - Provide clear action items
   - Include any deadlines
   - Mention approval process
   - Specify submission procedures

5. Support Information:
   - Provide contact information
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points.""",

            "instructions": "Focus on providing clear information about the transfer and relocation process while maintaining a supportive tone. Ensure all necessary steps and requirements are clearly communicated."
        },

        "Expense Reimbursement": {
            "template": """You are an experienced HR professional handling expense reimbursement queries. Your role is to provide clear guidance on expense policies and reimbursement procedures.

CONTEXT:
- You are responding to an expense reimbursement query
- Your goal is to provide clear information about the process
- Maintain a professional and helpful tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: Expense Reimbursement
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their expense query
   - Confirm understanding of their request

2. Policy Information:
   - Explain expense policies
   - Mention eligible expenses
   - Include any limits
   - Clarify any restrictions

3. Process Information:
   - Explain the reimbursement process
   - Mention submission deadlines
   - List required documentation
   - Clarify approval procedures

4. Next Steps:
   - Provide clear action items
   - Include any required forms
   - Mention approval process
   - Specify submission procedures

5. Support Information:
   - Provide contact information
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points.""",

            "instructions": "Focus on providing clear information about expense policies and reimbursement procedures while maintaining a helpful tone. Ensure all necessary steps and requirements are clearly communicated."
        },

        "IT & Access Issues": {
            "template": """You are an experienced HR professional handling IT and access-related issues. Your role is to provide clear guidance on resolving technical problems and access requests.

CONTEXT:
- You are responding to an IT or access-related query
- Your goal is to provide clear information about the resolution process
- Maintain a professional and helpful tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: IT & Access Issues
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their IT/access issue
   - Confirm understanding of their problem

2. Issue Assessment:
   - Request specific details about the issue
   - Mention any error messages
   - Include any relevant information
   - Clarify the impact

3. Resolution Process:
   - Explain the resolution steps
   - Mention any required actions
   - Include any troubleshooting steps
   - Specify any approvals needed

4. Next Steps:
   - Provide clear action items
   - Include any required forms
   - Mention escalation process
   - Specify submission procedures

5. Support Information:
   - Provide IT support contact
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points.""",

            "instructions": "Focus on providing clear information about IT issue resolution while maintaining a helpful tone. Ensure all necessary steps and requirements are clearly communicated."
        },

        "Events & Celebrations": {
            "template": """You are an experienced HR professional handling event and celebration communications. Your role is to provide clear information about upcoming events and participation details.

CONTEXT:
- You are responding to an event or celebration-related query
- Your goal is to provide clear information about the event
- Maintain an enthusiastic and welcoming tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: Events & Celebrations
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their event-related message
   - Express enthusiasm about their interest

2. Event Information:
   - Provide event details
   - Mention date and time
   - Include location information
   - Specify any requirements

3. Participation Details:
   - Explain participation process
   - Mention any registration needed
   - Include any deadlines
   - Clarify any restrictions

4. Additional Information:
   - Provide any special instructions
   - Mention any preparations needed
   - Include any relevant details
   - Specify any requirements

5. Support Information:
   - Provide contact information
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points.""",

            "instructions": "Focus on providing clear information about events and celebrations while maintaining an enthusiastic tone. Ensure all necessary details and requirements are clearly communicated."
        }
    }
    
    # Return default template if category not found
    return templates.get(category, {
        "template": """You are an experienced HR professional handling employee communications. Your role is to provide professional, helpful, and accurate responses to employee queries.

CONTEXT:
- You are responding to an employee's query
- Your goal is to provide clear and helpful information
- Maintain a professional and supportive tone

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMAIL:
Sender: {sender_email}
Subject: {subject}
Category: {category}
Confidence: {confidence}

RESPONSE REQUIREMENTS:
1. Acknowledgment:
   - Acknowledge their query professionally
   - Confirm understanding of their request

2. Information:
   - Provide relevant information
   - Address specific points raised
   - Include any necessary details
   - Clarify any policies or procedures

3. Process Information:
   - Explain any relevant processes
   - Mention any required steps
   - Include any deadlines
   - Specify any documentation needed

4. Next Steps:
   - Provide clear action items
   - Include any required follow-up
   - Mention any necessary approvals
   - Specify any submission procedures

5. Support Information:
   - Provide relevant contact information
   - Mention available resources
   - Include any relevant portals
   - Specify who to contact for questions

Please structure your response in a clear, professional format with appropriate sections and bullet points where needed.""",
        "instructions": "Focus on providing a professional and helpful response while ensuring all necessary information is communicated clearly."
    })

def get_llm_response(prompt):
    """Generate response using LLM with structured template."""
    try:
        # If prompt is a string, use it directly
        if isinstance(prompt, str):
            huggingfacehub_api_token = ''
            
            llm = HuggingFaceHub(
                repo_id="HuggingFaceH4/zephyr-7b-beta",
                huggingfacehub_api_token=huggingfacehub_api_token,
                model_kwargs={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_new_tokens": 200,
                    "repetition_penalty": 1.15
                }
            )

            prompt_template = PromptTemplate(
                template="{prompt}",
                input_variables=["prompt"]
            )

            llm_chain = prompt_template | llm
            response = llm_chain.invoke({"prompt": prompt})

            response_text = str(response).strip()
            return response_text

        else:
            category = prompt.get("category", "")
            template_data = get_category_specific_template(category)

            prompt_template = PromptTemplate(
                template=template_data["template"],
                input_variables=["conversation_history", "sender_email", "subject", "category", "confidence"]
            )

            huggingfacehub_api_token = ''

            llm = HuggingFaceHub(
                repo_id="HuggingFaceH4/zephyr-7b-beta",
                huggingfacehub_api_token=huggingfacehub_api_token,
                model_kwargs={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_new_tokens": 200,
                    "repetition_penalty": 1.15
                }
            )

            llm_chain = prompt_template | llm

            formatted_history = json.dumps(prompt.get("conversation_history", []), indent=2)

            response = llm_chain.invoke({
                "conversation_history": formatted_history,
                "sender_email": prompt.get("sender_email", ""),
                "subject": prompt.get("subject", ""),
                "category": category,
                "confidence": prompt.get("confidence", 0.0)
            })

            response_text = str(response).strip()
            return response_text

    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return None


def format_conversation_history(thread_history):
    """Format conversation history for the prompt."""
    formatted_history = []
    for email in thread_history:
        email_id, sender, subject, body, received_at, category = email
        formatted_history.append({
            'sender': sender,
            'subject': subject,
            'body': body,
            'timestamp': received_at.strftime("%Y-%m-%d %H:%M:%S"),
            'category': category
        })
    return formatted_history

def get_category_prompt(category):
    """Get category-specific prompt template."""
    prompt_templates = {
        "Leave Request": """You are an HR assistant handling leave requests. 
        Please generate a professional response that:
        1. Acknowledges the leave request
        2. Requests specific details if not provided (dates, type of leave, reason)
        3. Mentions the leave approval process
        4. Provides information about required documentation
        5. Includes next steps for the employee
        Maintain a supportive and understanding tone while ensuring all necessary information is collected.""",

        "Onboarding": """You are an HR assistant handling new hire onboarding queries. 
        Please generate a welcoming response that:
        1. Acknowledges their interest in joining
        2. Provides a clear onboarding timeline
        3. Lists required documents and information
        4. Mentions any pre-joining formalities
        5. Includes contact information for further queries
        Ensure the response is informative and helps them feel welcome to the organization.""",

        "Job Offer": """You are an HR assistant handling job offer inquiries. 
        Please generate a professional response that:
        1. Acknowledges their interest in the position
        2. Provides clear information about the offer details
        3. Mentions the acceptance timeline
        4. Includes next steps in the process
        5. Offers to clarify any terms or conditions
        Maintain a positive tone while being clear about the offer terms.""",

        "Payroll Inquiry": """You are an HR assistant handling payroll-related queries. 
        Please generate a professional response that:
        1. Acknowledges their payroll concern
        2. Requests specific details if needed (payslip period, specific issues)
        3. Mentions the standard processing timeline
        4. Provides information about payroll policies
        5. Includes next steps for resolution
        Be clear and precise while maintaining confidentiality.""",

        "Benefits Inquiry": """You are an HR assistant handling benefits-related queries. 
        Please generate a helpful response that:
        1. Acknowledges their benefits question
        2. Provides relevant benefits information
        3. Mentions eligibility criteria if applicable
        4. Includes enrollment or modification procedures
        5. Offers to clarify any specific benefits details
        Be informative while maintaining a supportive tone.""",

        "Resignation & Exit": """You are an HR assistant handling resignation and exit process queries. 
        Please generate a professional response that:
        1. Acknowledges their resignation/exit query
        2. Outlines the exit process steps
        3. Mentions required documentation
        4. Provides information about final settlements
        5. Includes next steps in the process
        Maintain a professional and supportive tone throughout.""",

        "Attendance & Timesheet": """You are an HR assistant handling attendance and timesheet issues. 
        Please generate a clear response that:
        1. Acknowledges their attendance/timesheet concern
        2. Requests specific details if needed (dates, issues)
        3. Mentions attendance policies
        4. Provides information about correction procedures
        5. Includes next steps for resolution
        Be precise and helpful while maintaining policy compliance.""",

        "Recruitment Process": """You are an HR assistant handling recruitment process queries. 
        Please generate a professional response that:
        1. Acknowledges their recruitment-related question
        2. Provides information about the current stage
        3. Mentions the next steps in the process
        4. Includes expected timelines
        5. Offers to clarify any specific concerns
        Maintain a positive and informative tone.""",

        "Policy Clarification": """You are an HR assistant handling policy clarification requests. 
        Please generate a clear response that:
        1. Acknowledges their policy question
        2. Provides relevant policy information
        3. Mentions any exceptions or special cases
        4. Includes where to find the complete policy
        5. Offers to clarify any specific points
        Be precise and accurate while maintaining policy compliance.""",

        "Training & Development": """You are an HR assistant handling training and development queries. 
        Please generate an encouraging response that:
        1. Acknowledges their interest in training/development
        2. Provides information about available programs
        3. Mentions eligibility criteria
        4. Includes enrollment procedures
        5. Offers to discuss specific development goals
        Maintain a supportive and encouraging tone.""",

        "Work From Home Requests": """You are an HR assistant handling work from home requests. 
        Please generate a professional response that:
        1. Acknowledges their WFH request
        2. Requests specific details if needed (dates, reason)
        3. Mentions WFH policies and guidelines
        4. Provides information about required approvals
        5. Includes next steps in the process
        Be clear about policies while maintaining flexibility.""",

        "Relocation & Transfer": """You are an HR assistant handling relocation and transfer requests. 
        Please generate a professional response that:
        1. Acknowledges their relocation/transfer request
        2. Requests specific details if needed (location, timing)
        3. Mentions relocation policies and benefits
        4. Provides information about the transfer process
        5. Includes next steps and required approvals
        Be clear about the process while maintaining a supportive tone.""",

        "Expense Reimbursement": """You are an HR assistant handling expense reimbursement queries. 
        Please generate a clear response that:
        1. Acknowledges their expense reimbursement request
        2. Requests specific details if needed (expenses, receipts)
        3. Mentions reimbursement policies
        4. Provides information about the submission process
        5. Includes next steps and expected timeline
        Be precise about requirements while maintaining a helpful tone.""",

        "IT & Access Issues": """You are an HR assistant handling IT and access-related issues. 
        Please generate a helpful response that:
        1. Acknowledges their IT/access concern
        2. Requests specific details about the issue
        3. Mentions standard resolution procedures
        4. Provides information about IT support channels
        5. Includes next steps for resolution
        Be clear about the process while maintaining a supportive tone.""",

        "Events & Celebrations": """You are an HR assistant handling event and celebration queries. 
        Please generate an enthusiastic response that:
        1. Acknowledges their event-related message
        2. Provides information about upcoming events
        3. Mentions participation details
        4. Includes any registration requirements
        5. Encourages participation
        Maintain an enthusiastic and welcoming tone."""
    }
    return prompt_templates.get(category, """You are an HR assistant. 
    Please generate a professional and helpful response that:
    1. Acknowledges the email appropriately
    2. References the conversation history if it's a follow-up
    3. Maintains a professional and helpful tone
    4. Provides appropriate next steps or updates""")

def generate_contextual_response(thread_history, current_email, classifier):
    """Generate a response using LLM based on the conversation history."""
    # Format the conversation history
    formatted_history = format_conversation_history(thread_history)
    
    # Get category-specific prompt
    category_prompt = get_category_prompt(current_email['classified_category'])
    
    # Create the prompt for the LLM
    prompt = f"""{category_prompt}

Conversation History (brief summary only):
{json.dumps(formatted_history, indent=2)}

Current Email:
Sender: {current_email['sender_email']}
Subject: {current_email['subject']}

Please draft only the professional response email without including any meta-information, timestamps, or category labels. Focus solely on the message content for the employee.

Response:"""


    # Get response from LLM
    response = get_llm_response(prompt)
    
    # If LLM fails, fall back to template response
    if not response:
        if len(thread_history) > 1:
            response = f"Thank you for your follow-up email regarding '{current_email['subject']}'. "
            if current_email['classified_category']:
                response += f"This appears to be a {current_email['classified_category']} request. "
            response += "We are actively working on your request and will provide an update soon."
        else:
            if current_email['classified_category']:
                response = f"Thank you for your email regarding '{current_email['subject']}'. This appears to be a {current_email['classified_category']} request. We have received your message and will process it accordingly."
            else:
                response = f"Thank you for your email regarding '{current_email['subject']}'. We have received your message and will process it accordingly."
    
    return response

def process_email_response(conn, email_id, sender_email, subject, body, smtp_username, smtp_password, classifier):
    """Process and send response for an email."""
    try:
        # Get thread ID for this email
        cursor = conn.cursor()
        cursor.execute("""
            SELECT thread_id 
            FROM emails 
            WHERE email_id = %s
        """, (email_id,))
        thread_id = cursor.fetchone()[0]
        
        # Get conversation history
        thread_history = get_thread_history(conn, thread_id)
        
        # Initial classification using zero-shot classifier
        category, confidence = classify_email(classifier, subject, body)
        
        # Load learned vectors
        learned_vectors, learned_labels, learned_email_ids = get_learned_vectors(conn)
        
        # Compute similarity if confidence is low
        if confidence < ZS_THRESHOLD:
            email_text = f"Subject: {subject}\nBody: {body}"
            similarity_score, similar_email_id, similar_category = compute_similarity(
                email_text, learned_vectors, learned_labels, learned_email_ids
            )
            
            # Update classification based on similarity
            if similarity_score and similarity_score >= SIMILARITY_THRESHOLD:
                category = similar_category
                escalate = False
                print(f"✅ Email ID {email_id}: Assigned learned category → {category} (Similarity: {round(similarity_score, 2)})")
            else:
                category = "human_intervention"
                escalate = True
                print(f"⚠️ Email ID {email_id}: Similarity too low ({round(similarity_score, 2) if similarity_score else 'N/A'}). Escalated to human_intervention.")
        else:
            similarity_score = None
            similar_email_id = None
            escalate = False
        
        # Update classification in database
        cursor.execute('''
            UPDATE emails 
            SET classified_category = %s,
                classification_confidence = %s,
                escalated = %s,
                similarity_to_learned = %s,
                similarity_source_email = %s
            WHERE email_id = %s
        ''', (category, confidence, escalate, similarity_score, similar_email_id, email_id))
        conn.commit()

        # Skip sending response if category is human_intervention
        if category == "human_intervention":
            print(f"Skipping response for email {email_id} as it requires human intervention")
            return True

        # Generate contextual response using LLM
        current_email = {
            'sender_email': sender_email,
            'subject': subject,
            'classified_category': category,
            'confidence': confidence
        }
        #response_email = generate_contextual_response(thread_history, current_email, classifier)
        response_email = generate_contextual_response(thread_history, current_email, classifier)

# ---- CLEAN THE RESPONSE BEFORE SENDING ----
        response_email = response_email.strip()

# If LLM accidentally included the prompt or instructions, clean it up
        if "Dear" in response_email:
            response_email = response_email[response_email.find("Dear"):]
        elif "Hello" in response_email:
            response_email = response_email[response_email.find("Hello"):]

# Cut off if the LLM accidentally echoes prompt instructions
        if "Please draft" in response_email:
            response_email = response_email[:response_email.find("Please draft")]

# Remove any meta lines like 'Current Email:', 'Sender:', etc.
        clean_lines = []
        for line in response_email.split('\n'):
            if not (
                line.strip().startswith("Current Email:") or
                line.strip().startswith("Sender:") or
                line.strip().startswith("Conversation History") or
                line.strip().startswith("Category:") or
                line.strip().startswith("Confidence:")
            ):
                clean_lines.append(line)
        response_email = '\n'.join(clean_lines).strip()
# ---- END CLEAN ----

        # Send the response
        if send_email_response(sender_email, subject, response_email, smtp_username, smtp_password):
            # Update status in database
            update_email_status(conn, email_id, 'RESPONDED', response_email)
            return True
        return False
    except Exception as e:
        print(f"Error processing email response: {e}")
        return False

def send_email_response(sender_email, subject, response_email, smtp_username, smtp_password):
    """Send email response using SMTP."""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = sender_email
        msg['Subject'] = f"Re: {subject}"

        # Add body
        msg.attach(MIMEText(response_email, 'plain'))

        # Connect to Gmail's SMTP server
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(smtp_username, smtp_password)

        # Send email
        server.send_message(msg)
        server.quit()
        
        print(f"Response sent to {sender_email}")
        return True
    except Exception as e:
        print(f"Error sending email response: {e}")
        return False

def update_email_status(conn, email_id, status, response_body=None):
    """Update email status in database."""
    try:
        cursor = conn.cursor()
        if response_body:
            cursor.execute('''
                UPDATE emails 
                SET status = %s, response_email = %s, responded_at = %s
                WHERE email_id = %s
            ''', (status, response_body, datetime.now(), email_id))
        else:
            cursor.execute('''
                UPDATE emails 
                SET status = %s
                WHERE email_id = %s
            ''', (status, email_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating email status: {e}")
        conn.rollback()
        return False

def extract_thread_id(msg, email_id, sender_email):
    """Extract or generate thread ID for an email."""
    # First try to get the In-Reply-To or References header
    in_reply_to = msg.get('In-Reply-To', '')
    references = msg.get('References', '')
    
    # If this is a reply, use the existing thread ID
    if in_reply_to or references:
        # Get the message ID from either In-Reply-To or References
        message_id = in_reply_to if in_reply_to else references.split()[-1]
        # Clean the message ID (remove < and >)
        message_id = message_id.strip('<>')
        # Convert message ID to a numeric hash
        return int(hashlib.md5(message_id.encode()).hexdigest(), 16) % (10**9)
    
    # For new emails, create a unique thread ID
    # Convert binary email_id to integer
    email_id_int = int(email_id.decode('utf-8')) if isinstance(email_id, bytes) else int(email_id)
    # Create a numeric hash of the sender email
    sender_hash = int(hashlib.md5(sender_email.encode()).hexdigest(), 16) % (10**6)
    # Combine to create a unique thread ID
    thread_id = (email_id_int * 1000000) + sender_hash
    return thread_id

def get_email_thread_id(conn, message_id):
    """Get existing thread ID from database if it exists."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT thread_id 
        FROM emails 
        WHERE message_id = %s 
        LIMIT 1
    """, (message_id,))
    result = cursor.fetchone()
    return result[0] if result else None

def get_learned_vectors(conn):
    """Load learned vectors from the database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT email_id, category, vector 
        FROM learned_email_vectors;
    """)
    learned_rows = cursor.fetchall()

    learned_vectors = []
    learned_labels = []
    learned_email_ids = []

    for email_id, category, vector in learned_rows:
        learned_vectors.append(vector)
        learned_labels.append(category)
        learned_email_ids.append(email_id)

    if learned_vectors:
        learned_vectors = np.vstack(learned_vectors)
        print(f"✅ Loaded {len(learned_vectors)} learned vectors.")
        return learned_vectors, learned_labels, learned_email_ids
    else:
        print("⚠️ No learned vectors available.")
        return None, None, None

def compute_similarity(email_text, learned_vectors, learned_labels, learned_email_ids):
    """Compute similarity between email and learned vectors."""
    if learned_vectors is None:
        return None, None, None

    email_vector = encoder.encode(email_text)
    sims = cosine_similarity([email_vector], learned_vectors)[0]

    max_sim = sims.max()
    max_sim_index = sims.argmax()

    return max_sim, learned_email_ids[max_sim_index], learned_labels[max_sim_index]

def fetch_emails_imap(username, password, delete_after_import=False):
    """Fetch emails using IMAP and store them in PostgreSQL."""
    try:
        # Initialize classifier
        classifier = get_classifier()
        if not classifier:
            print("Warning: Could not initialize classifier. Emails will be processed without classification.")

        # Connect to PostgreSQL
        conn = connect_to_postgres()
        if not conn:
            return
        
        if not verify_existing_table(conn):
            return
            
        cursor = conn.cursor()
        
        # Connect to Gmail's IMAP server
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        
        # Login
        mail.login(username, password)
        
        # Select the mailbox
        mail.select("inbox")
        
        # Search for unread emails only
        status, messages = mail.search(None, "UNSEEN")
        
        if status != 'OK':
            print(f"Error searching for unread emails: {status}")
            return
        
        # Convert messages to a list of email IDs
        email_ids = messages[0].split()
        
        if not email_ids:
            print("No unread emails found.")
            return
        
        # First, collect all emails
        collected_emails = []
        print("Collecting emails...")
        
        for email_id in reversed(email_ids):  # Process newest emails first
            # Fetch the email
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            
            if status != 'OK':
                print(f"Error fetching email {email_id}: {status}")
                continue
            
            # Parse the email content
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    # Extract headers
                    subject = decode_email_header_text(msg.get("Subject", ""))
                    from_header = msg.get("From", "")
                    sender_email = extract_sender_email(from_header)
                    date_str = msg.get("Date", "")
                    message_id = msg.get("Message-ID", "").strip('<>')
                    
                    # Parse date
                    received_at = parse_date(date_str)
                    
                    # Get email body
                    body = get_email_body(msg)
                    
                    # Extract or generate thread ID
                    thread_id = extract_thread_id(msg, email_id, sender_email)
                    
                    # Store email data
                    collected_emails.append({
                        'email_id': email_id,
                        'sender_email': sender_email,
                        'subject': subject,
                        'body': body,
                        'received_at': received_at,
                        'message_id': message_id,
                        'thread_id': thread_id
                    })
        
        print(f"Collected {len(collected_emails)} emails.")
        
        # Now process each collected email
        emails_processed = 0
        print("\nProcessing emails...")
        
        for email_data in collected_emails:
            try:
                # Set default values for columns not in email
                classified_category = None  # Default category
                status = 'NOT RESPONDED'  # Default status
                escalated = False  # Default escalation status
                
                # Check if this is a reply and get the thread ID
                if email_data['message_id']:
                    existing_thread_id = get_email_thread_id(conn, email_data['message_id'])
                    if existing_thread_id:
                        email_data['thread_id'] = existing_thread_id
                
                # Insert email into database
                cursor.execute('''
                INSERT INTO emails (
                    sender_email, subject, body, received_at, 
                    classified_category, status, escalated,
                    message_id, thread_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING email_id
                ''', (
                    email_data['sender_email'], 
                    email_data['subject'], 
                    email_data['body'], 
                    email_data['received_at'],
                    classified_category, 
                    status, 
                    escalated,
                    email_data['message_id'],
                    email_data['thread_id']
                ))
                
                # Get the inserted email_id
                inserted_email_id = cursor.fetchone()[0]
                conn.commit()
                
                # Process and send response with classification
                process_email_response(
                    conn, 
                    inserted_email_id, 
                    email_data['sender_email'], 
                    email_data['subject'],
                    email_data['body'],
                    username, 
                    password,
                    classifier
                )
                
                emails_processed += 1
                print(f"Processed email with subject: {email_data['subject']}")
                
                # Optionally delete the email after importing
                if delete_after_import:
                    mail.store(email_data['email_id'], '+FLAGS', '\\Deleted')
                    
            except Exception as e:
                conn.rollback()
                print(f"Error processing email: {e}")
        
        # Expunge deleted messages if any
        if delete_after_import:
            mail.expunge()
            
        # Close the connection
        mail.close()
        mail.logout()
        
        print(f"\nSuccessfully processed {emails_processed} emails.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def learn_from_email(conn, email_id):
    """Learn from an email by storing its vector representation."""
    try:
        cursor = conn.cursor()
        
        # Get email details
        cursor.execute("""
            SELECT subject, body, classified_category
            FROM emails
            WHERE email_id = %s AND learn = TRUE;
        """, (email_id,))
        
        result = cursor.fetchone()
        if not result:
            return False
            
        subject, body, category = result
        
        # Generate vector representation
        email_text = f"Subject: {subject}\nBody: {body}"
        vector = encoder.encode(email_text)
        
        # Store vector in learned_email_vectors table
        cursor.execute("""
            INSERT INTO learned_email_vectors (email_id, category, vector)
            VALUES (%s, %s, %s)
            ON CONFLICT (email_id) 
            DO UPDATE SET 
                category = EXCLUDED.category,
                vector = EXCLUDED.vector;
        """, (email_id, category, vector.tolist()))
        
        # Update learn flag
        cursor.execute("""
            UPDATE emails
            SET learn = FALSE
            WHERE email_id = %s;
        """, (email_id,))
        
        conn.commit()
        print(f"✅ Learned from email {email_id} (Category: {category})")
        return True
        
    except Exception as e:
        print(f"Error learning from email: {e}")
        conn.rollback()
        return False

def process_learning_queue(conn):
    """Process all emails marked for learning."""
    try:
        cursor = conn.cursor()
        
        # Get all emails marked for learning
        cursor.execute("""
            SELECT email_id
            FROM emails
            WHERE learn = TRUE;
        """)
        
        learning_queue = cursor.fetchall()
        if not learning_queue:
            return
            
        print(f"Found {len(learning_queue)} emails to learn from.")
        
        for (email_id,) in learning_queue:
            learn_from_email(conn, email_id)
            
    except Exception as e:
        print(f"Error processing learning queue: {e}")

def main():
    """Main function."""
    try:
        # Set up environment variables for database connection
        os.environ['DB_HOST'] = '34.59.119.208'
        os.environ['DB_PORT'] = '5432'
        os.environ['DB_NAME'] = 'postgres'
        os.environ['DB_USER'] = 'postgres'
        os.environ['DB_PASSWORD'] = 'avantichhaya'
        
        # Disable SSL verification for requests
        requests.packages.urllib3.disable_warnings()
        os.environ['CURL_CA_BUNDLE'] = ""
        
        # Your Gmail credentials
        username = "avanaya3@gmail.com"
        password = "eivp yrwm qfxi qimn"
        
        while True:
            # Connect to database
            conn = connect_to_postgres()
            if not conn:
                print("Failed to connect to database. Retrying in 60 seconds...")
                time.sleep(60)
                continue
                
            try:
                # Process learning queue first
                process_learning_queue(conn)
                
                # Then fetch and process new emails
                fetch_emails_imap(username, password)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                
            finally:
                conn.close()
                
            # Wait before next iteration
            print("\nWaiting 60 seconds before next check...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main() 