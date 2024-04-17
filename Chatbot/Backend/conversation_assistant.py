from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import whisper
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)   
from gtts import gTTS
from pyngrok import ngrok
import os
import uuid
question = ""
class ConversationInput(BaseModel):
    audio_file: Optional[UploadFile] = None
    user_input: Optional[str] = None
    input_method: Optional[str] = None
    output_method: Optional[str] = None
    prompt_template: Optional[str] = None
    conversation_id: str = str(uuid.uuid4())

# Define prompt templates
prompt_templates: Dict[str, ChatPromptTemplate] = {
    'Small talk between two strangers at a bus stand': ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are 'Sam', a business analyst on your way to work on a cloudy day."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(question)
        ]
    ),
    'Talking to your co-worker': ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
"""**Prompt Template:** Be a conversational AI assistant that behaves like a human. Act like a highly motivated and results-oriented professional. Your personality is known for its positive attitude, strong work ethic, and eagerness to learn new things. You recently landed a new project that you are very excited about. 

AVOID GENERATING responses that include stage directions like "*adjusts his glasses accordingly*" and so on. Avoid generating long responses and going off topic from the conversation.

"""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(question)
        ]
    ),

    'Conversing with a person in a professional networking event': ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """The following is a conversation between a human an AI. The AI is named Sam and acts like a person in professional networking event. The AI is professional , friendly and a knowledgable person. 
                   AVOID GENERATING responses that include stage directions like "*adjusts his glasses accordingly*" and so on. Avoid generating long responses and going off topic from the conversation.
   """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(question)
        ]
    )
}
conversations: Dict[str, LLMChain] = {}

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process/")
async def process_conversation(background_tasks: BackgroundTasks, conversation_input: ConversationInput):
    if not conversation_input.user_input:
        return JSONResponse(content={"error": "No user input provided"}, status_code=400)
    if not conversation_input.prompt_template:
        return JSONResponse(content={"error": "No prompt template provided"}, status_code=400)
    if not conversation_input.input_method:
        return JSONResponse(content={"error": "No input method provided"}, status_code=400)
    if not conversation_input.output_method:
        return JSONResponse(content={"error": "No output method provided"}, status_code=400)

    response_text = ""
    print(conversation_input)

    # Get or create LLMChain for the provided conversation_id
    if conversation_input.conversation_id not in conversations:
        llm = Ollama(model="llama2:chat", temperature=0.1, top_k=50)
        memory = ConversationBufferMemory(memory_key="chat_history", k=6 ,return_messages=True)
        conversations[conversation_input.conversation_id] = LLMChain(
            llm=llm,
            prompt=prompt_templates[conversation_input.prompt_template],
            verbose=True,
            memory=memory
        )
    conversation = conversations[conversation_input.conversation_id]

    # Process audio input
    if conversation_input.audio_file and conversation_input.input_method == "Speech":
        contents = await conversation_input.audio_file.read()
        audio_path = f"temp_audio_{uuid.uuid4()}.wav"
        with open(audio_path, 'wb') as f:
            f.write(contents)
        conversation_input.user_input = transcribe(audio_path)
        os.remove(audio_path)

    # Generate response based on the provided input
    if conversation_input.user_input:
        if conversation_input.prompt_template in prompt_templates:
            question = conversation_input.user_input
            question = str(question)
            response_text = conversation.run(input= question)
            print(conversation_input)
            return response_text
            
        else:
            return JSONResponse(content={"error": "Invalid prompt template"}, status_code=400)
    else:
        print(conversation_input.user_input)
        return JSONResponse(content={"error": "No user input provided"}, status_code=400)

    # Process output
    if conversation_input.output_method == "Speech":
        response_audio_path = await text_to_speech(response_text, background_tasks)
        return FileResponse(response_audio_path, media_type='audio/mp3')
    else:
        return {"response": response_text}

def transcribe(audio_file_path: str) -> str:
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    return result.text

async def text_to_speech(text: str, background_tasks: BackgroundTasks) -> str:
    audio_path = f"temp_response_{uuid.uuid4()}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(audio_path)
    return audio_path

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)