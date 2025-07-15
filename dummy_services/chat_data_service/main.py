"""Dummy Chat Data Service for testing QA Analysis Service."""
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class ChatTranscript(BaseModel):
    """Model for chat transcript data."""
    
    transcript_id: str = Field(..., description="Unique identifier for the transcript")
    transcript: str = Field(..., description="The chat transcript content")
    agent_id: str = Field(..., description="Agent who handled the chat")
    customer_id: str = Field(..., description="Customer identifier")
    timestamp: str = Field(..., description="Chat timestamp")


app = FastAPI(
    title="Dummy Chat Data Service",
    description="Provides sample chat transcripts for QA analysis testing",
    version="1.0.0"
)

# Sample chat transcripts for testing
SAMPLE_TRANSCRIPTS = [
    ChatTranscript(
        transcript_id="chat_001",
        transcript="""
Agent: Hello! Welcome to Comm100 support. How can I help you today?
Customer: Hi, I'm having trouble setting up email notifications in my account.
Agent: I'd be happy to help you with that. Can you tell me what specific issue you're experiencing?
Customer: I can't find where to enable email notifications for new tickets.
Agent: Sure! To enable email notifications, go to Settings > Notifications > Email Settings. Then check the box for "New Ticket Notifications".
Customer: I don't see that option in my Settings menu.
Agent: Let me check your account permissions. What's your role in the organization?
Customer: I'm an administrator.
Agent: As an admin, you should see that option. Try refreshing your browser and clearing your cache, then check again.
Customer: That worked! I can see it now. Thank you so much!
Agent: You're welcome! Is there anything else I can help you with today?
Customer: No, that's all. Thanks again!
Agent: Have a great day!
        """.strip(),
        agent_id="agent_001",
        customer_id="customer_001",
        timestamp="2024-01-15T10:30:00Z"
    ),
    ChatTranscript(
        transcript_id="chat_002",
        transcript="""
Agent: Good morning! This is Sarah from Comm100 support. How may I assist you?
Customer: Hi Sarah, I need help with integrating our CRM with Comm100.
Agent: I'll be glad to help with that integration. Which CRM system are you looking to integrate?
Customer: We use Salesforce. Is that supported?
Agent: Yes, Comm100 has a robust Salesforce integration. You can find it in the Apps marketplace within your admin panel.
Customer: How do I access the Apps marketplace?
Agent: Go to Admin > Apps & Integrations > Marketplace, then search for "Salesforce".
Customer: I found it! Do I need special permissions to install it?
Agent: You'll need admin privileges to install integrations. Do you have admin access?
Customer: Yes, I'm the account admin.
Agent: Perfect! Click "Install" and follow the setup wizard. You'll need your Salesforce API credentials.
Customer: Great, I'll do that right now. Thanks for your help!
Agent: You're very welcome! Feel free to reach out if you need any assistance during the setup process.
        """.strip(),
        agent_id="agent_002", 
        customer_id="customer_002",
        timestamp="2024-01-15T11:45:00Z"
    ),
    ChatTranscript(
        transcript_id="chat_003",
        transcript="""
Agent: Hello! Welcome to Comm100. I'm Mike, how can I help you today?
Customer: Hi Mike, my chat widget isn't appearing on my website.
Agent: I'm sorry to hear that. Let me help you troubleshoot this issue. Can you provide your website URL?
Customer: Sure, it's www.example-company.com
Agent: Thank you. First, can you confirm you've installed the chat code on your website?
Customer: Yes, I copied the code from the installation guide.
Agent: Great. The code should be placed right before the closing </body> tag. Can you verify that's where you placed it?
Customer: Let me check... Actually, I think I put it in the <head> section instead.
Agent: That would cause the issue. The chat widget code needs to be placed just before the </body> tag to load properly.
Customer: I'll move it there right now.
Agent: Perfect! After you move it, clear your browser cache and refresh the page. The widget should appear.
Customer: It's working now! Thank you so much for your quick help.
Agent: Excellent! I'm glad we got that resolved quickly. Is there anything else I can help you with?
Customer: No, that's everything. Thanks again!
Agent: You're welcome! Have a wonderful day.
        """.strip(),
        agent_id="agent_003",
        customer_id="customer_003", 
        timestamp="2024-01-15T14:20:00Z"
    )
]


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Dummy Chat Data Service", "status": "running"}


@app.get("/transcripts", response_model=List[ChatTranscript])
async def get_transcripts(limit: Optional[int] = 10):
    """Get a list of chat transcripts."""
    return SAMPLE_TRANSCRIPTS[:limit] if limit else SAMPLE_TRANSCRIPTS


@app.get("/transcripts/{transcript_id}", response_model=ChatTranscript)
async def get_transcript(transcript_id: str):
    """Get a specific chat transcript by ID."""
    for transcript in SAMPLE_TRANSCRIPTS:
        if transcript.transcript_id == transcript_id:
            return transcript
    
    raise HTTPException(status_code=404, detail="Transcript not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 