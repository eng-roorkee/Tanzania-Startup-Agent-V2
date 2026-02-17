import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Ruaha LLM integration APi")

# NOTE
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url ="https://api.groq.com/openai/v1"
)

class PromptRequest(BaseModel):
    message: str

class PromptResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=PromptResponse)
async def generate_chat_response(request: PromptRequest):
    try:
        response = client.chat.completions.create( 
            model="llama-3.3-70b-versatile",
            messages=[  # Correction 1: 'messages' instead of 'message'
                {"role": "system", "content": "You are a helpful, rigorous and intelligent academic assistant"},
                {"role": "user", "content": request.message}
            ]
        )
        llm_reply = response.choices[0].message.content 
        return PromptResponse(reply=llm_reply)  # Correction 2: explicit 'reply=' keyword
    except Exception as e: 
        # By printing the error to the terminal, we can see exactly what went wrong next time
        print(f"Error occurred: {e}") 
        raise HTTPException(status_code=500, detail=str(e))