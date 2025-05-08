from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
CORS(app)

# ✅ Database connection
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    host="34.59.119.208",
    password="avantichhaya"
)
cursor = conn.cursor()

# Email configuration
SMTP_USERNAME = "avanaya3@gmail.com"
SMTP_PASSWORD = "eivp yrwm qfxi qimn"

def send_email_response(sender_email, subject, response_body):
    """Send email response using SMTP."""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = sender_email
        msg['Subject'] = f"Re: {subject}"

        # Add body
        msg.attach(MIMEText(response_body, 'plain'))

        # Connect to Gmail's SMTP server
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SMTP_USERNAME, SMTP_PASSWORD)

        # Send email
        server.send_message(msg)
        server.quit()
        
        print(f"Response sent to {sender_email}")
        return True
    except Exception as e:
        print(f"Error sending email response: {e}")
        return False

#########################################
# 📌 GET: Fetch escalated emails only
#########################################
@app.route("/api/escalations", methods=["GET"])
def get_escalations():
    cursor.execute("""
        SELECT email_id, sender_email, subject, body, received_at, classified_category, learn
        FROM emails
        WHERE classified_category = 'human_intervention'
    """)
    rows = cursor.fetchall()

    # Convert to list of dicts
    data = []
    for row in rows:
        data.append({
            "email_id": row[0],
            "sender_email": row[1],
            "subject": row[2],
            "body": row[3],
            "received_at": row[4].strftime('%Y-%m-%d %H:%M:%S'),
            "classified_category": row[5],
            "learn": row[6]  # can be None, True, or False
        })

    return jsonify(data)

#########################################
# 📌 POST: Update category and learn value
#########################################
@app.route("/api/update_email", methods=["POST"])
def update_email():
    data = request.json
    email_id = data.get("email_id")
    updated_category = data.get("updated_category")
    response_text = data.get("response")
    learn = data.get("learn")  # True/False

    try:
        # First get the sender's email and subject
        cursor.execute("""
            SELECT sender_email, subject
            FROM emails
            WHERE email_id = %s
        """, (email_id,))
        result = cursor.fetchone()
        if not result:
            return jsonify({"error": "Email not found"}), 404
            
        sender_email, subject = result

        # Send the response email
        if response_text:
            email_sent = send_email_response(sender_email, subject, response_text)
            if not email_sent:
                return jsonify({"error": "Failed to send email response"}), 500

        # Update category, learn, response, and status
        cursor.execute("""
            UPDATE emails
            SET classified_category = %s,
                learn = %s,
                response_email = %s,
                status = 'RESPONDED',
                responded_at = %s
            WHERE email_id = %s
        """, (updated_category, learn, response_text, datetime.datetime.now(), email_id))

        conn.commit()

        print(f"✅ Updated email {email_id} → Category: {updated_category} | Learn: {learn}")
        print(f"📝 Response saved and sent to: {sender_email}")

        return jsonify({"message": "Email updated and response sent successfully!"})

    except Exception as e:
        conn.rollback()
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500

#########################################
# ✅ Run Flask
#########################################
if __name__ == "__main__":
    app.run(debug=True)
