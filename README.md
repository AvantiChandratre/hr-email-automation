# HR Email Response Automation System

An **end-to-end email processing system** designed to automate HR-related email handling, including retrieval, classification, response generation, and continuous learning.

---

##  Overview

This project aims to streamline HR communications by automating email management. The system automatically classifies incoming emails, generates context-aware responses using advanced NLP models, and escalates complex emails for manual review, significantly reducing administrative workload.

---

##  Key Features

- **Automatic Email Fetching** (IMAP)
- **Email Classification** using Hugging Face’s (`facebook/bart-large-mnli`)
- **Semantic Similarity** via SentenceTransformer (`all-mpnet-base-v2`)
- **Contextual Automated Responses** (LangChain + Hugging Face Zephyr-7b-beta)
- **Structured Storage** (PostgreSQL)
- **Frontend Dashboard** (React)
- **SMTP Integration** (Automated Email Sending)
- **Adaptive Learning Loop** (Using Semantic Similarity)

---

## Tech Stack

- **Backend:** Python, Hugging Face Transformers, LangChain
- **Database:** PostgreSQL
- **Frontend:** React (Vite, TailwindCSS, Shadcn UI)
- **Machine Learning Models:** 
  - Classification: `facebook/bart-large-mnli`
  - Similarity: SentenceTransformer (`all-mpnet-base-v2`)
  - Response Generation: Zephyr-7b-beta via LangChain
- **Protocols:** IMAP (fetching), SMTP (sending)
- **Cloud Services:** Google Cloud PostgreSQL
- **Security:** SSL/TLS encryption, secure environment variables

---

##  System Architecture

```plaintext
Gmail (IMAP)
    │
    ├── Email Parsing
    │     └─ PostgreSQL (Storage)
    │
    ├── Email Classification & Similarity Check
    │     └─ NLP Models (Zero-shot + Semantic Vectors)
    │
    ├── Response Generation (LLM via LangChain)
    │     └─ Custom Prompt Templates
    │
    ├── Response Dispatch (SMTP)
    │
    └── Frontend Dashboard (React)
          └─ Monitoring & Manual Overrides
```
---

## Database Schema

### Emails Table

| Column                     | Description                                       |
|----------------------------|---------------------------------------------------|
| `email_id`                 | Unique identifier for each email                   |
| `sender_email`             | Sender's email address                            |
| `subject`                  | Subject line of the email                         |
| `body`                     | Content of the email                              |
| `received_at`              | Timestamp when the email was received             |
| `classified_category`      | Predicted category by classification model        |
| `classification_confidence`| Confidence score of the prediction                |
| `status`                   | Status (`responded` or `pending`)                 |
| `escalated`                | Whether manual intervention is required (Boolean) |
| `response_email`           | Generated automated response                      |
| `thread_id`                | Identifier linking email conversations            |
| `learn`                    | Flag indicating email marked for learning (Boolean)|
| `similarity_source_email`  | Email ID of the most similar prior email          |
| `similarity_to_learned`    | Similarity score to previous learned emails       |
| `message_id`               | Unique message ID from email header               |
| `responded_at`             | Timestamp when response was sent                  |

### Learned Vectors Table

| Column       | Description                           |
|--------------|---------------------------------------|
| `email_id`   | Associated email's unique identifier  |
| `category`   | Category of the email                 |
| `vector`     | Semantic vector embedding             |
| `created_at` | Timestamp when vector was generated   |

---

## Business Benefits

- **Efficiency**: Reduces manual email handling and administrative workload.
- **Consistency and Accuracy**: Ensures professional and error-free responses.
- **Scalability**: Easily manages higher email volumes without increased overhead.
- **Continuous Learning**: Automatically adapts and improves with new data.
- **Auditability**: Comprehensive logging enables transparency and compliance.

---

## Development Approach

- **Iterative Development**: Agile methodology with incremental improvements.
- **Prompt Engineering**: Tailored templates for accurate NLP responses.
- **Threshold Optimization**: Carefully tuned thresholds for balancing automation and manual review.
- **Robust Security**: Secure handling and storage of sensitive information.

---

## Frontend (React Dashboard)

The frontend dashboard enables HR teams to:

- Monitor incoming emails and their automated responses.
- Manage escalation and manual review processes.
- Flag emails for inclusion in adaptive learning models.

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/yourusername/hr-email-automation.git
cd hr-email-automation

