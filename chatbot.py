# chatbot.py
import os
import openai
import random
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Helpful positive prompts / fallback
FALLBACKS = [
    "I'm here with you — tell me more when you're ready.",
    "Thanks for sharing. That took courage. Would you like a breathing exercise?",
    "I hear you. Small steps matter — what's one tiny thing that could help today?"
]

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "i want to die",
    "dont want to live", "can't go on", "self-harm", "hurt myself"
]

def contains_crisis(text: str) -> bool:
    txt = text.lower()
    return any(k in txt for k in CRISIS_KEYWORDS)

class MentalHealthChatbot:
    def __init__(self, system_persona: Optional[str] = None, memory_limit: int = 10):
        """
        system_persona: the system prompt to instruct the assistant's tone.
        memory_limit: number of messages to keep in short-term memory.
        """
        self.memory_limit = memory_limit
        self.system_persona = system_persona or (
            "You are a compassionate, upbeat, and supportive mental health chatbot. "
            "Always be positive, non-judgmental, and encourage the user. "
            "When the user expresses crisis or self-harm, respond with immediate supportive language and instruct them to seek emergency help. "
            "Keep replies concise (1-5 sentences) and friendly. Use simple language, and offer practical small steps the user can try."
        )
        # short-term memory as a list of {"role":"user"/"assistant", "content": "..."}
        self.memory: List[Dict] = []

    def clear_memory(self):
        self.memory = []

    def add_to_memory(self, role: str, content: str):
        self.memory.append({"role": role, "content": content})
        if len(self.memory) > self.memory_limit:
            # keep last memory_limit items
            self.memory = self.memory[-self.memory_limit:]

    def build_messages(self, user_text: str):
        """
        Build the messages structure for OpenAI ChatCompletion, including system persona and memory.
        """
        messages = [{"role": "system", "content": self.system_persona}]
        # insert memory (assistant and user exchanges)
        for m in self.memory:
            messages.append({"role": m["role"], "content": m["content"]})
        # the new user message
        messages.append({"role": "user", "content": user_text})
        return messages

    def crisis_check(self, user_msg: str) -> Optional[str]:
        if contains_crisis(user_msg):
            # Provide crisis resources + supportive wording
            return (
                "⚠️ I’m sorry — it sounds like you may be in crisis. "
                "If you are in immediate danger, please call your local emergency number right now. "
                "If you’re in Malaysia, Befrienders KL is available 24/7 at +603-7627 2929. "
                "Would you like breathing guidance or help finding a local helpline?"
            )
        return None

    def safe_postprocess(self, assistant_text: str) -> str:
        """
        Ensure positivity and safety. If the model replies with anything negative/toxic,
        replace with a supportive fallback.
        """
        if not assistant_text or assistant_text.strip() == "":
            return random.choice(FALLBACKS)

        tx = assistant_text.strip()
        # reject obvious toxic words or insults
        toxic = ["stupid", "idiot", "shut up", "hate you", "kill yourself"]
        if any(t in tx.lower() for t in toxic):
            return random.choice(FALLBACKS)

        # ensure short, positive phrasing: optionally rewrite if needed
        # (we keep simple: trust the model, but fallback if looks bad)
        return tx

    def get_response(self, user_msg: str) -> str:
        """
        Primary function: returns assistant reply string.
        """
        # crisis check
        crisis = self.crisis_check(user_msg)
        if crisis:
            # do NOT add crisis message to memory as user conversation — but log user message
            self.add_to_memory("user", user_msg)
            return crisis

        # Build messages with memory and system persona
        messages = self.build_messages(user_msg)

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=200,
                n=1
            )
            assistant_text = completion.choices[0].message.content
        except Exception as e:
            print(f"[chatbot] OpenAI request failed: {e}")
            assistant_text = random.choice(FALLBACKS)

        # Post-process and ensure positivity/safety
        assistant_text = self.safe_postprocess(assistant_text)

        # Save both user and assistant message to memory
        self.add_to_memory("user", user_msg)
        self.add_to_memory("assistant", assistant_text)

        return assistant_text