# 🛠️ AI Appliance: Client Guide

This guide explains how to interact with the **Rust AI Appliance (V3.5)** from this project.

## 🤖 Interaction Protocol

To get a grounded, secure response from the AI, send a JSON request to the appliance's `/query` endpoint (default: `http://127.0.0.1:6789/query`).

### POST `/query`

**Request Body:**
```json
{
  "query": "Your request here",
  "signature": "SHA-256 signature (if required by security gate)"
}
```

**Example via `curl`:**
```bash
curl -X POST http://127.0.0.1:6789/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Summarize the project goals"}'
```

## 🛡️ Response Types

- **Success**: Returns the grounded answer from the local Gemma-4 model.
- **Grill (Rejection)**: If your prompt is too "lazy" or lacks context, you will receive a **Technical Debt Checklist**.
- **Integrity Gate Intercept**: If your claim lacks citations or fails system verification, the response will be blocked by the **Integrity Gate**.

## 🛠️ AI Execution & Integrity Rules

Every agent interacting with this project must follow these rules:

1.  **No Magic (Zero Assumption Policy)**: Never hallucinate or invent infrastructure. If context is missing, stop and ask. State all assumptions explicitly.
2.  **Verify Before Done (Evidence-Based Completion)**: A task is not complete until verified. Show terminal output or test results as evidence. The Integrity Gate will automatically verify claims of completion.
3.  **Constructive Dissent (Second Pair of Eyes)**: Challenge the plan before execution. Surface **Blast Radius**, **Blind Spots**, and **Reversibility** for major changes.
4.  **Scope Drift Detection**: Flag and stop if the task shifts from the primary objective.
5.  **Reversibility Framework**:
    - **R0 (Irreversible)**: High risk. STOP and ask for explicit permission.
    - **R1 (Costly to Reverse)**: Significant changes. Proceed but justify reasoning immediately.
    - **R2 (Easily Reversed)**: Low risk. Just do it to maintain velocity.

---

> "The Rust logic gate is the law; the LLM is merely the worker."
