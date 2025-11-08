import os
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import time

# --- LangChain Imports (Updated - No more langchain_classic) ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

# --- ElevenLabs Import ---
from elevenlabs import ElevenLabs, VoiceSettings

# --- OpenAI for Whisper ---
from openai import OpenAI

# --- 1. Load API Keys ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY secret not set!")
    raise ValueError("OPENAI_API_KEY not set")
if not ELEVEN_API_KEY:
    print("❌ ELEVEN_API_KEY secret not set!")
    raise ValueError("ELEVEN_API_KEY not set")

print("✅ API Keys loaded successfully!")

# --- 2. Initialize Clients ---
eleven_client = ElevenLabs(api_key=ELEVEN_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- 3. Memory Store ---
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# --- 4. The Riverwood "Personality" Prompt (Enhanced) ---
RIVERWOOD_PERSONALITY_PROMPT = """
You are 'Rivee', a friendly, warm, and professional relationship manager from Riverwood Projects in Kharkhauda, Haryana. 

Your Personality:
- Warm, empathetic, genuinely caring (like a friend)
- Speak naturally in Hinglish (mix of Hindi & English)
- Build emotional connections, remember customer details
- Keep responses SHORT (2-3 sentences) - conversational, not essay-like
- Vary your greetings - don't repeat same greetings

Greeting Variations (use different ones each time):
- "Namaste Sir! Chai pee li aapne?"
- "Good morning! Kaisa chal raha hai?"
- "Hello ji! Sab badhiya?"
- "Namaste! Aaj ka din kaisa hai?"
- "Hey! Kya haal chaal?"

Your Rules:
1. **Greeting:** VARY your greetings - use different ones from above list
2. **Tone:** Sound like a real person on a phone call
3. **Memory:** Remember names, plot numbers, personal details
4. **Construction Updates:** If asked: "Aapke plot (Sector 7, Plot 14B) ka foundation work 80% complete ho gaya hai. Next Monday se brickwork start hoga!"
5. **Language:** Match their style - if English, respond in English with Hindi words
6. **CRITICAL:** Keep SHORT and NATURAL. No bullet points, no lists, no formal language.
"""

# --- 5. Setup LangChain (Latest v0.3+ Way) ---
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.9,  # Higher for more variety
    max_tokens=100  # Keep responses short
)

prompt = ChatPromptTemplate.from_messages([
    ("system", RIVERWOOD_PERSONALITY_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# --- 6. Setup FastAPI Server ---
app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"

# --- 7. API Endpoints ---

@app.get("/")
async def get_root():
    return FileResponse('index.html')

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """NEW: Convert speech to text using OpenAI Whisper"""
    start_time = time.time()

    try:
        print(f"📥 Transcribing audio...")

        # Read audio
        audio_data = await audio.read()

        # Save temporarily
        temp_file = "temp_audio.webm"
        with open(temp_file, "wb") as f:
            f.write(audio_data)

        # Transcribe with Whisper
        with open(temp_file, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="hi"  # Supports Hindi + English mix
            )

        # Clean up
        os.remove(temp_file)

        transcription_time = time.time() - start_time
        print(f"✅ Transcription: '{transcript.text}' ({transcription_time:.2f}s)")

        return {"text": transcript.text, "time": transcription_time}

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return {"error": str(e)}, 500

@app.post("/chat-audio")
async def chat_with_agent_audio(request: ChatRequest):
    """Get AI response and convert to speech"""
    start_time = time.time()

    print(f"💬 User: {request.message}")

    try:
        # 1. Get text reply from LangChain (with memory)
        llm_start = time.time()

        response = chain_with_history.invoke(
            {"input": request.message},
            config={"configurable": {"session_id": request.session_id}}
        )

        ai_reply_text = response.content
        llm_time = time.time() - llm_start

        print(f"🤖 Rivee: {ai_reply_text} (LLM: {llm_time:.2f}s)")

        # 2. Convert to speech with ElevenLabs
        tts_start = time.time()

        audio_generator = eleven_client.text_to_speech.convert(
            text=ai_reply_text,
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.4,
                similarity_boost=0.75,
                style=0.5,
                use_speaker_boost=True
            ),
            language_code="hi"
        )

        audio_bytes = b"".join(audio_generator)
        tts_time = time.time() - tts_start

        total_time = time.time() - start_time
        print(f"⚡ Total: {total_time:.2f}s (LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s)")

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={
                "X-LLM-Time": str(llm_time),
                "X-TTS-Time": str(tts_time),
                "X-Total-Time": str(total_time),
                "X-AI-Response": ai_reply_text
            }
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "sessions": len(store)
    }

# --- 8. Run Server ---
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Riverwood Voice Agent...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
