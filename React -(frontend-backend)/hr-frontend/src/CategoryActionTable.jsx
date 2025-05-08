
import React, { useState } from 'react';

const initialStateFlowData = [
    { category: "Leave Request", action: "Auto-approve leave" },
    { category: "Onboarding", action: "Send onboarding checklist" },
    { category: "Job Offer", action: "Forward to HR manager" },
    { category: "Payroll Inquiry", action: "Forward to payroll team" },
    { category: "Benefits Inquiry", action: "Forward to benefits coordinator" },
    { category: "Resignation & Exit", action: "Forward to HR manager" },
    { category: "Attendance & Timesheet", action: "Send attendance guidelines" },
    { category: "Recruitment Process", action: "Send interview schedule link" },
    { category: "Policy Clarification", action: "Send HR policy document" },
    { category: "Training & Development", action: "Send training program schedule" },
    { category: "Work From Home Requests", action: "Send remote work policy" },
    { category: "Relocation & Transfer", action: "Forward to HR transfer team" },
    { category: "Expense Reimbursement", action: "Send reimbursement form" },
    { category: "IT & Access Issues", action: "Forward to IT support" },
    { category: "Events & Celebrations", action: "Forward to events coordinator" },
    { category: "human_intervention", action: "No auto-response (await admin review)" }
];

const availableActions = [
    "Auto-approve leave",
    "Send onboarding checklist",
    "Forward to HR manager",
    "Forward to payroll team",
    "Forward to benefits coordinator",
    "Send attendance guidelines",
    "Send interview schedule link",
    "Send HR policy document",
    "Send training program schedule",
    "Send remote work policy",
    "Forward to HR transfer team",
    "Send reimbursement form",
    "Forward to IT support",
    "Forward to events coordinator"
];

function CategoryActionTable() {
    const [stateFlowData, setStateFlowData] = useState(initialStateFlowData);
    const [editedAction, setEditedAction] = useState("");

    const handleActionChange = (index, newAction) => {
        const updatedData = [...stateFlowData];
        updatedData[index].action = newAction;
        setStateFlowData(updatedData);
    };

    return (
        <div>
            <h2>State Flow Editor</h2>
            <table border="1" cellPadding="10">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {stateFlowData.map((row, index) => (
                        <tr key={index}>
                            <td>{row.category}</td>
                            <td>
                                {row.category === "human_intervention" ? (
                                    <select
                                        value={row.action}
                                        onChange={(e) => handleActionChange(index, e.target.value)}
                                    >
                                        <option value="">-- Select Action --</option>
                                        {availableActions.map((actionOption, idx) => (
                                            <option key={idx} value={actionOption}>
                                                {actionOption}
                                            </option>
                                        ))}
                                    </select>
                                ) : (
                                    row.action
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

export default CategoryActionTable;
