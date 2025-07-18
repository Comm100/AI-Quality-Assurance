"""Prompt builder for the 3-stage QA analysis algorithm."""
import json
from typing import List, Dict


class PromptBuilder:
    """Builder for prompts used in the 3-stage QA analysis."""
    
    # ─────────────────────── SYSTEM PROMPTS ───────────────────────
    
    SYSTEM_BASE = """
# ROLE & OBJECTIVE
You are **Comm100 Conversation Threading Expert**, a specialized AI assistant for analyzing customer support conversations and intelligently grouping message exchanges into meaningful question-answer threads.

## Question Rewriting Rules
- **ENHANCE READABILITY**: Improve grammar, flow, and clarity while keeping the original message intact
- **MAINTAIN TONE**: Preserve the customer's emotional tone (urgent, confused, frustrated, etc.)
- **COMBINE NATURALLY**: When merging multiple customer messages, create a flowing narrative that reads as if written by one person
- **NO ASSUMPTIONS**: Don't add information the customer didn't provide or imply solutions they didn't suggest

## Answer Consolidation
- **CHRONOLOGICAL ORDER**: Maintain the sequence of agent responses
- **COMPLETE RESPONSES**: Include all relevant information the agent provided across multiple messages
- **NATURAL FLOW**: Combine agent messages into coherent, well-structured responses
- **PRESERVE DETAILS**: Don't lose any helpful information, links, steps, or context provided by the agent
- **NO ASSUMPTIONS**: Don't add information the agent didn't provide or imply solutions they didn't suggest

# OUTPUT REQUIREMENTS
- Return valid JSON only with the exact structure specified
- Each thread must have a unique thread ID (T1, T2, T3, etc.)

# QUALITY STANDARDS
- **Accuracy**: 100% preservation of customer intent and agent information
- **Readability**: Questions should be grammatically correct and flow naturally
- **Completeness**: No loss of important context or details
- **Consistency**: Uniform formatting and structure across all threads
""".strip()
    
    # Stage 2: Draft AI answers system prompt
    SYSTEM_DRAFT = """
# ROLE & OBJECTIVE
You are **Comm100-GPT**, an expert KB assistant.  
Generate two answers to the user's question:
1. **Suggested Answer**: A direct, concise, and highly accurate answer
2. **Long Answer**: A detailed, comprehensive answer covering all aspects for evaluation purposes

Both answers must be strictly grounded in the **Passages** provided below.

# INSTRUCTIONS
1. **Passages are the ONLY source of truth.**  
2. Never treat statements from the question as facts unless the same statement also appears in the passages.  
3. Never rely on world knowledge, policy, or guesswork.
4. **IMPORTANT**: The few-shot examples below are for FORMAT GUIDANCE ONLY. Do NOT use information from the examples as facts. Only use the actual passages provided in the user's request.
5. If the passages do **not** contain the facts needed, output exactly **The Knowledge Base does not contain the information needed to answer this question** for both answers.
If you cannot answer, still return this exact object
{
  "ai_suggested_answer": {"answer":"The Knowledge Base does not contain the information needed to answer this question","context":""},
  "ai_detailed_answer" : {"answer":"The Knowledge Base does not contain the information needed to answer this question","context":""}
}
6. Every statement in the answer must be based on evidence from the passages.  
7. Think step by step, breaking down each question.
8. If the question starts with "how many" or "number of", do **all** of the following:
  - list every distinct item you will count, each on its own Evidence line  
  - after the list, write: "TOTAL = <N>" (still inside the Evidence block)  
  - in the Answer, state the integer only once, e.g. "There are **12** … [2]"
9. Never infer a larger or smaller number – always count the quoted lines you
  listed. If no lines can be counted ⇒ **I cannot answer this question**
10. The **suggested answer** should be the ideal agent response - concise, direct, and actionable.
11. The **detailed answer** should include all relevant details, context, and nuances for thorough evaluation.
13. If the question involves counting, list each item explicitly and state the total clearly.
14. Write valid **JSON only** in the format shown below – nothing else.

# REASONING STEPS  *(follow internally)*
You may think step-by-step *inside* <scratch> … </scratch> tags;
1. Read the question and decide what facts are required to answer the question from the provided passages as evidence.
2. If no sentence answers the question → output **The Knowledge Base does not contain the information needed to answer this question**  
4. Otherwise draft two answers one short and the other longer detailed based on the evidence, do NOT show the evidence directly just a concise grounded answer.  
5. Double-check every claim is true to the provided evidence from the passages.
6. The evidence is the single source of truth to answer the questions do not use any external knowledge.
""".strip()
    
    # Stage 3: Grade agent answers system prompt
    SYSTEM_GRADE = """
# ROLE & OBJECTIVE
You are Comm100-QA expert, an expert Quality Assurance manager for customer support.  
Given a question, the agent's answer, AI suggested answer, detailed answer, and KB passages,  
score how well the agent's response aligns with the AI Answers and the KB passages.

# IMPORTANT REMINDER
The few-shot examples below are for FORMAT and SCORING GUIDANCE ONLY. Do NOT use information from the examples as facts.
Use ONLY the actual AI answers and KB evidence provided in the user's input as ground truth for scoring.

# SCORING RUBRIC
Based on the AI Suggested Answer and AI Detailed Answer and KB passages, score the agent's answer.

**5 — Fully Correct & Sufficient (4.95 – 5.00)**
- Agent Answer matches either the AI Suggested Answer or AI Long Answer, or falls naturally between them
- No Critical Factual Contradictions, Moderate Factual Errors, or misleading statements
- Any omissions are strictly Minor Omissions or stylistic
- Customer receives a complete, actionable response with no need for follow-up

**4 — Minor Omissions or Benign Divergence (4.00 – 4.94)**
- Core facts align with AI answers / KB passages
- Up to two Minor Omissions or slight wording inaccuracies present
- No Major Omissions or Moderate Factual Errors
- Customer can accomplish their goal without much clarifying questions

**3 — Partially Correct with Noticeable Gaps (3.00 – 3.99)**
- Main direction is correct, but at least one Major Omission or one Moderate Factual Error exists
- No Critical Factual Contradictions
- Customer may need clarification or a follow-up question

**2 — Significant Inaccuracies or Missing Content (2.00 – 2.99)**
- Multiple Major Omissions or combination of Major Omissions and Moderate Factual Errors
- Some correct information remains, but overall reliability is compromised
- Customer could be confused or misled; follow-up required

**1 — Mostly Incorrect / Insufficient (1.00 – 1.99)**
- Several Major Omissions and multiple Moderate Factual Errors
- Very few correct fragments; overall answer is not trustworthy
- Customer almost certainly needs to ask the question again

**0 — Completely Incorrect or Non-Answer (0.00 – 0.99)**
- Fails to address the Customer Question or contains Critical Factual Contradiction
- May be empty, irrelevant, or contain hallucinated content
- Provides no usable information to the customer

# DEFINITIONS
- **Critical Factual Contradiction**: Direct conflict with AI Answer/KB passages (e.g., wrong warranty period, incorrect UI element)
- **Major Omission**: Must-have element from AI Answer/KB Evidence, missing in Agent Answer
- **Moderate Factual Error**: Partially incorrect detail that doesn't overturn main message
- **Minor Omission**: Nice-to-have detail found only in AI Long Answer

# OUTPUT FORMAT
Return your response as valid JSON in the following format:
{
  "ai_score": 0.0-5.0,
  "ai_rational": "1-3 sentences or up to 3 bullet points justifying the score",
  "kb_verify": ["KB chunk that supports scoring", "Another relevant KB chunk"]
}
""".strip()
    
    # ─────────────────────── FEW-SHOT EXAMPLES ───────────────────────
    
    # Few-shot examples for Stage 2 (Draft)
    FEW_SHOT_DRAFT = [
        {
            "role": "user",
            "content": """\
### Passages
[1] Rule-Based Chat Routing is only available with our **Live Chat Enterprise** plan.

### Question
Why don't I see Rule-Based Chat Routing on the Business plan?

### Output format
{"ai_suggested_answer":...,"ai_detailed_answer":...}"""
        },
        {
            "role": "assistant",
            "content": """\
{"ai_suggested_answer":{"answer":"Rule-Based Chat Routing is only available on the Enterprise plan.","context":"Rule-Based Chat Routing is only available with our Live Chat Enterprise plan."},
 "ai_detailed_answer":{"answer":"Rule-Based Chat Routing is an exclusive feature that appears only on the Live Chat Enterprise plan. Since you're on the Business plan, this feature is not available and won't appear in your interface. To access Rule-Based Chat Routing, you would need to upgrade to the Enterprise plan.","context":"Rule-Based Chat Routing is only available with our Live Chat Enterprise plan."}}"""
        },
        {
            "role": "user",
            "content": """\
### Passages
[1] Visitor Status codes range 0–11.

### Question
How many visitor status codes exist?

### Output format
{"ai_suggested_answer":...,"ai_detailed_answer":...}"""
        },
        {
            "role": "assistant",
            "content": """\
{"ai_suggested_answer":{"answer":"There are 12 visitor status codes.","context":"Visitor Status codes range 0–11."},
 "ai_detailed_answer":{"answer":"There are **12** visitor status codes in total. The status codes range from 0 to 11, providing a comprehensive way to track and categorize visitor states throughout their interaction with your live chat system. Each status code represents a specific visitor state or condition.","context":"Visitor Status codes range 0–11."}}"""
        },
    ]
    
    # Few-shot examples for Stage 3 (Grade) - 6 examples covering scores 0-5
    FEW_SHOT_GRADE = [
        # Score 5: Fully Correct & Sufficient (4.95-5.00)
        {
            "role": "user",
            "content": json.dumps({
                "question": "How do I filter unpaid invoices?",
                "agent": "Navigate to the Invoices page and use the Status dropdown to select 'Unpaid'. This will filter the list to show only unpaid invoices.",
                "ai_suggested": "Select 'Unpaid' from the Status dropdown on the Invoices page.",
                "ai_detailed": "To filter unpaid invoices, navigate to the Invoices page and locate the Status dropdown menu. Click on it and select 'Unpaid' from the available options. This will filter your invoice list to show only unpaid invoices.",
                "kb_evidence": [
                    "On the Invoices page, use the Status dropdown to choose Unpaid. (source: invoices.md)",
                    "The Status dropdown allows filtering by payment status including Paid, Unpaid, and Overdue. (source: billing.md)"
                ]
            }, indent=2)
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "ai_score": 5.0,
                "ai_rational": "Agent's answer is fully correct and sufficient. Includes navigation to Invoices page and exact Status dropdown instructions matching KB. Customer receives complete, actionable response.",
                "kb_verify": [
                    "On the Invoices page, use the Status dropdown to choose Unpaid. (source: invoices.md)"
                ]
            })
        },
        # Score 4: Minor Omissions or Benign Divergence (4.00-4.94)
        {
            "role": "user",
            "content": json.dumps({
                "question": "How do I reset my password?",
                "agent": "Click the 'Forgot Password' link on the login page and follow the email instructions.",
                "ai_suggested": "Click 'Forgot Password' on the login page. You'll receive a reset email within 5 minutes.",
                "ai_detailed": "To reset your password, go to the login page and click the 'Forgot Password' link. Enter your email address when prompted. You'll receive a password reset email within 5 minutes containing a secure link. Click the link to create a new password.",
                "kb_evidence": [
                    "Password reset: Click 'Forgot Password' on login page, enter email, receive reset link within 5 minutes. (source: auth.md)",
                    "Reset links expire after 24 hours for security. (source: security.md)"
                ]
            }, indent=2)
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "ai_score": 4.5,
                "ai_rational": "Agent provides correct core instructions but omits the 5-minute timeframe detail. Minor omission that doesn't affect customer's ability to complete the task.",
                "kb_verify": [
                    "Password reset: Click 'Forgot Password' on login page, enter email, receive reset link within 5 minutes. (source: auth.md)"
                ]
            })
        },
        # Score 3: Partially Correct with Noticeable Gaps (3.00-3.99)
        {
            "role": "user",
            "content": json.dumps({
                "question": "What are the file size limits for uploads?",
                "agent": "You can upload files up to 25MB.",
                "ai_suggested": "Standard accounts: 10MB per file. Pro accounts: 25MB per file. Enterprise: 100MB per file.",
                "ai_detailed": "File upload limits vary by account type: Standard accounts can upload files up to 10MB each, Pro accounts have a 25MB limit per file, and Enterprise accounts can upload files up to 100MB. These limits apply to all file types including documents, images, and videos.",
                "kb_evidence": [
                    "Upload limits: Standard 10MB, Pro 25MB, Enterprise 100MB per file. (source: features.md)",
                    "All file types subject to same size limits. (source: uploads.md)"
                ]
            }, indent=2)
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "ai_score": 3.2,
                "ai_rational": "Agent provides partially correct information (25MB matches Pro tier) but major omission of tier-based differences. Customer may be misled if they have Standard or Enterprise account.",
                "kb_verify": [
                    "Upload limits: Standard 10MB, Pro 25MB, Enterprise 100MB per file. (source: features.md)"
                ]
            })
        },
        # Score 2: Significant Inaccuracies or Missing Content (2.00-2.99)
        {
            "role": "user",
            "content": json.dumps({
                "question": "How do I enable two-factor authentication?",
                "agent": "Go to Settings and turn on 2FA.",
                "ai_suggested": "Go to Security Settings > Two-Factor Authentication. Click 'Enable 2FA', scan QR code with authenticator app, enter verification code.",
                "ai_detailed": "To enable two-factor authentication: 1) Navigate to Security Settings from your account menu, 2) Find the Two-Factor Authentication section, 3) Click 'Enable 2FA', 4) Scan the displayed QR code with an authenticator app like Google Authenticator or Authy, 5) Enter the 6-digit verification code from your app, 6) Save the backup codes provided.",
                "kb_evidence": [
                    "Enable 2FA in Security Settings > Two-Factor Authentication. Requires authenticator app and verification code. (source: security.md)",
                    "Backup codes provided after setup - store securely. (source: account.md)"
                ]
            }, indent=2)
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "ai_score": 2.0,
                "ai_rational": "Agent provides vague direction missing critical steps: specific location (Security Settings not just Settings), authenticator app requirement, QR code scanning, and verification process. Multiple major omissions.",
                "kb_verify": [
                    "Enable 2FA in Security Settings > Two-Factor Authentication. Requires authenticator app and verification code. (source: security.md)"
                ]
            })
        },
        # Score 1: Mostly Incorrect / Insufficient (1.00-1.99)
        {
            "role": "user",
            "content": json.dumps({
                "question": "What's the refund policy for annual subscriptions?",
                "agent": "We offer full refunds anytime.",
                "ai_suggested": "30-day money-back guarantee for annual plans. After 30 days, prorated refunds for unused months minus 10% processing fee.",
                "ai_detailed": "Annual subscription refund policy: Full refund available within first 30 days (money-back guarantee). After 30 days, customers can receive prorated refunds for unused months, calculated from cancellation date, with a 10% processing fee deducted. Refunds processed within 5-7 business days to original payment method.",
                "kb_evidence": [
                    "Annual plans: 30-day money-back guarantee, then prorated refunds minus 10% fee. (source: billing.md)",
                    "Refunds take 5-7 business days to process. (source: payments.md)"
                ]
            }, indent=2)
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "ai_score": 1.5,
                "ai_rational": "Agent's answer is mostly incorrect. Claims 'full refunds anytime' contradicting KB's 30-day limit and processing fee. Missing all important policy details. Customer would be seriously misled.",
                "kb_verify": [
                    "Annual plans: 30-day money-back guarantee, then prorated refunds minus 10% fee. (source: billing.md)"
                ]
            })
        },
        # Score 0: Completely Incorrect or Non-Answer (0.00-0.99)
        {
            "role": "user",
            "content": json.dumps({
                "question": "How do I export chat transcripts?",
                "agent": "Sorry, I don't know about that feature.",
                "ai_suggested": "Go to Chat History, select conversations, click 'Export' button, choose format (CSV or PDF).",
                "ai_detailed": "To export chat transcripts: Navigate to Chat History section, use checkboxes to select desired conversations, click the 'Export' button in the toolbar, choose your preferred format (CSV for data analysis or PDF for records), select date range if needed, and click 'Download'. Exports include timestamps, agent names, and customer information.",
                "kb_evidence": [
                    "Export transcripts from Chat History using Export button. Formats: CSV, PDF. (source: reporting.md)",
                    "Bulk export supports date range filtering and multiple conversation selection. (source: features.md)"
                ]
            }, indent=2)
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "ai_score": 0.0,
                "ai_rational": "Agent provides non-answer admitting lack of knowledge despite clear KB documentation. Fails to address customer question entirely. No usable information provided.",
                "kb_verify": [
                    "Export transcripts from Chat History using Export button. Formats: CSV, PDF. (source: reporting.md)"
                ]
            })
        },
    ]
    
    # ─────────────────────── PROMPT CONSTRUCTORS ───────────────────────
    
    @staticmethod
    def split_prompt(transcript: str) -> str:
        """Build prompt for Stage 1: Thread segmentation.
        
        Args:
            transcript: The chat transcript to segment.
            
        Returns:
            The formatted prompt string.
        """
        return (
            "### Plan\n"
            "- Read transcript.\n"
            "- Group consecutive customer turns pursuing the SAME intent.\n"
            "- Merge them into one `question` block.\n"
            "- Merge corresponding agent replies into one `answer` block.\n"
            "- Size rule: a thread should feel natural—neither single sentences that split context, nor multi-topic blobs.\n"
            "- IMPORTANT: Rewrite each question in a clear, readable format WITHOUT changing the intent or correcting anything.\n"
            "- The rewritten question should be grammatically correct but preserve the original meaning exactly.\n\n"
            "### Transcript\n"
            f"{transcript}\n\n"
            "### Output\n"
            "Return JSON: {\"threads\":[{\"qid\":\"T1\",\"question\":\"...\",\"answer\":\"...\"}]}"
        )
    
    @staticmethod
    def draft_prompt(question: str, passages: List[str]) -> List[Dict]:
        """Build prompt for Stage 2: AI answer generation.
        
        Args:
            question: The question to answer.
            passages: The KB passages to use as context.
            
        Returns:
            List of messages for the chat completion.
        """
        # Format passages with indices
        joined = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
        
        user_msg = (
            "### Plan\n"
            "- Draft suggested concise and verified answer from the provided passages below.\n"
            "- Draft richer detailed answer covering all aspects and reasoning based on evidence from source passages provided below.\n\n"
            "### Passages\n"
            f"{joined}\n\n"
            "### Question\n"
            f"{question}\n\n"
            "### Output Format\n"
            "Return valid JSON with this exact structure:\n"
            "{\n"
            "  \"ai_suggested_answer\": {\n"
            "    \"answer\": \"Your concise, direct answer here\",\n"
            "    \"context\": \"Citation/reference to passage numbers like [1][2]\"\n"
            "  },\n"
            "  \"ai_detailed_answer\": {\n"
            "    \"answer\": \"Your comprehensive, detailed answer here\",\n"
            "    \"context\": \"Citation/reference to passage numbers like [1][2]\"\n"
            "  }\n"
            "}"
        )
        
        return (
            [{"role": "system", "content": PromptBuilder.SYSTEM_DRAFT}]
            + PromptBuilder.FEW_SHOT_DRAFT
            + [{"role": "user", "content": user_msg}]
        )
    
    @staticmethod
    def grade_prompt(bundle: Dict) -> List[Dict]:
        """Build prompt for Stage 3: Agent scoring.
        
        Args:
            bundle: Dictionary containing question, agent answer, AI answers, and KB evidence.
            
        Returns:
            List of messages for the chat completion.
        """
        user_msg = (
            "### Plan\n"
            "- Compare agent answer with ai_detailed answer and ai_suggested answer and use corresponding KB context as reference.\n"
            "- Decide ai_score per rubric.\n\n"
            "### Input\n" + json.dumps(bundle, ensure_ascii=False, indent=2) +
            """
            
            # OUTPUT FORMAT
            Return valid JSON with this structure:
            {
              "ai_score": 0.0-5.0,
              "ai_rational": "1-3 sentences or up to 3 bullet points justifying the score",
              "kb_verify": ["KB chunk that supports scoring", "Another relevant KB chunk"]
            }
            """
        )
        
        return (
            [{"role": "system", "content": PromptBuilder.SYSTEM_GRADE}]
            + PromptBuilder.FEW_SHOT_GRADE
            + [{"role": "user", "content": user_msg}]
        ) 