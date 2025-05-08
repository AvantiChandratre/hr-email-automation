import React, { useState, useEffect } from 'react';
import axios from 'axios';

const categories = [
    "Leave Request", "Onboarding", "Job Offer", "Payroll Inquiry", "Benefits Inquiry",
    "Resignation & Exit", "Attendance & Timesheet", "Recruitment Process", "Policy Clarification",
    "Training & Development", "Work From Home Requests", "Relocation & Transfer", "Expense Reimbursement",
    "IT & Access Issues", "Events & Celebrations", "human_intervention"
];

function EscalationTable() {
    const [emails, setEmails] = useState([]);

    useEffect(() => {
        fetchEscalations();
    }, []);

    const fetchEscalations = () => {
        axios.get('http://127.0.0.1:5000/api/escalations')
            .then(response => {
                const updatedEmails = response.data.map(email => ({
                    ...email,
                    updated_category: email.classified_category,
                    response: '',
                    learn: false
                }));
                setEmails(updatedEmails);
            })
            .catch(error => console.error('Error fetching escalations:', error));
    };

    const handleCategoryChange = (email_id, value) => {
        setEmails(emails.map(email =>
            email.email_id === email_id ? { ...email, updated_category: value } : email
        ));
    };

    const handleResponseChange = (email_id, value) => {
        setEmails(emails.map(email =>
            email.email_id === email_id ? { ...email, response: value } : email
        ));
    };

    const handleLearnChange = (email_id, value) => {
        setEmails(emails.map(email =>
            email.email_id === email_id ? { ...email, learn: value === 'Yes' } : email
        ));
    };

    const handleSave = (email_id) => {
        const email = emails.find(e => e.email_id === email_id);
        axios.post('http://127.0.0.1:5000/api/update_email', {
            email_id: email.email_id,
            updated_category: email.updated_category,
            response: email.response,
            learn: email.learn
        })
        .then(() => {
            alert(`Updated email ID ${email_id} successfully!`);
            fetchEscalations();  // Refresh table
        })
        .catch(error => {
            console.error('Error updating email:', error);
            alert('Failed to update email.');
        });
    };

    return (
        <div>
            <h2>Escalated Emails</h2>
            <table border="1" cellPadding="5">
                <thead>
                    <tr>
                        <th>Email ID</th>
                        <th>Sender</th>
                        <th>Subject</th>
                        <th>Body</th>
                        <th>Received At</th>
                        <th>Updated Category</th>
                        <th>Response</th>
                        <th>Learn?</th>
                        <th>Save</th>
                    </tr>
                </thead>
                <tbody>
                    {emails.map(email => (
                        <tr key={email.email_id}>
                            <td>{email.email_id}</td>
                            <td>{email.sender_email}</td>
                            <td>{email.subject}</td>
                            <td>{email.body}</td>
                            <td>{email.received_at}</td>
                            <td>
                                <select value={email.updated_category} onChange={e => handleCategoryChange(email.email_id, e.target.value)}>
                                    {categories.map(cat => (
                                        <option key={cat} value={cat}>{cat}</option>
                                    ))}
                                </select>
                            </td>
                            <td>
                                <input
                                    type="text"
                                    value={email.response}
                                    onChange={e => handleResponseChange(email.email_id, e.target.value)}
                                />
                            </td>
                            <td>
                                <select value={email.learn ? "Yes" : "No"} onChange={e => handleLearnChange(email.email_id, e.target.value)}>
                                    <option value="Yes">Yes</option>
                                    <option value="No">No</option>
                                </select>
                            </td>
                            <td>
                                <button onClick={() => handleSave(email.email_id)}>Save</button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

export default EscalationTable;
