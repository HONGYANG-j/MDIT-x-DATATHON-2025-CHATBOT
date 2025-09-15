# chatbot.py - Advanced Natural Conversational Mental Health Chatbot (OpenAI API version, improved)
import os
import random
import re
import sqlite3
import logging
from openai import OpenAI
from dotenv import load_dotenv   # ✅ add this

# ------------------- Load Environment -------------------
load_dotenv()  # ✅ this reads your .env file

# ------------------- Setup Logging -------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("chatbot-api")

# ------------------- OpenAI Setup -------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------- Database Memory -------------------
DB_FILE = "chat_memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            user TEXT,
            role TEXT,
            msg TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_message(user, role, msg):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memory (user, role, msg) VALUES (?, ?, ?)", (user, role, msg))
    conn.commit()
    conn.close()

def load_memory(user, limit=8):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, msg FROM memory WHERE user=? ORDER BY rowid DESC LIMIT ?", (user, limit))
    rows = cursor.fetchall()
    conn.close()
    return rows[::-1]  # chronological order

# ------------------- Chatbot Core -------------------
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "i want to die",
    "dont want to live", "can't go on", "self-harm", "hurt myself",
    "end it all", "worthless", "no reason to live", "give up"
]

POSITIVE_REINFORCEMENTS = [
    "That’s really brave of you to share 💙",
    "I’m proud of you for opening up 🌸",
    "Even small steps forward matter 🙏",
    "You’re stronger than you realize 💪",
    "It’s okay to take things one step at a time 🌟"
]

class AdvancedAPIChatbot:
    def __init__(self):
        self.user_name = None
        self.mood = "neutral"

    def analyze_mood(self, text: str) -> str:
        txt = text.lower()
        if any(k in txt for k in CRISIS_KEYWORDS):
            return "crisis"
        elif any(w in txt for w in ["sad", "depressed", "anxious", "stress", "tired", "angry"]):
            return "negative"
        elif any(w in txt for w in ["happy", "good", "great", "excited", "better", "relieved"]):
            return "positive"
        return "neutral"

    def get_crisis_response(self):
        return (
            "💙 I hear how much pain you're in, and I’m really concerned.\n\n"
            "This is serious, and you deserve immediate support:\n"
            "📞 Befrienders KL: +603-7627 2929 (24/7)\n"
            "📞 Talian Kasih: 15999 (24/7 mental health line)\n"
            "📞 Emergency: 999 (if you're in danger)\n\n"
            "You’re not alone. Please reach out to them 🙏"
        )

    def detect_name(self, message: str):
        patterns = [r"(?:my name is|i'm|call me) (\w+)", r"(?:name's|name is) (\w+)"]
        for p in patterns:
            match = re.search(p, message.lower())
            if match:
                return match.group(1).capitalize()
        return None

    def humanize(self, response: str) -> str:
        if not response:
            return "I'm here with you 💙 What’s been on your mind?"

        # Remove robotic phrasing
        ai_phrases = ["as an AI", "according to", "research shows", "as a language model"]
        for phrase in ai_phrases:
            response = response.replace(phrase, "")

        # Add natural fillers and soft tone
        starters = ["Honestly, ", "You know, ", "Well, ", "Hmm, ", "Yeah, "]
        if random.random() < 0.25 and not response.startswith(tuple(starters)):
            response = random.choice(starters) + response[0].lower() + response[1:]

        # Sprinkle emojis to soften tone
        if random.random() < 0.4:
            response += random.choice([" 😊", " 💙", " 🌸", " 🙏", " 🌟", " 🤗"])

        return response.strip()

    def summarize_memory(self, history):
        """Summarize old messages into a compact note to preserve context."""
        if len(history) < 4:
            return history
        summary_text = "Summary of earlier chat: "
        summary_text += " | ".join([f"{role}: {msg}" for role, msg in history[:-3]])
        return [("system", summary_text)] + history[-3:]

    def generate(self, user: str, user_msg: str):
        # Crisis detection
        mood = self.analyze_mood(user_msg)
        if mood == "crisis":
            save_message(user, "user", user_msg)
            return self.get_crisis_response()

        # Update mood tracking
        self.mood = mood

        # Name detection
        if not self.user_name:
            name = self.detect_name(user_msg)
            if name:
                self.user_name = name
                save_message(user, "user", user_msg)
                save_message(user, "assistant", f"Hey {name}! 😊 Nice to meet you. How’s your day?")
                return f"Hey {name}! 😊 Nice to meet you. How’s your day?"

        # Load memory and summarize if too long
        history = load_memory(user, limit=8)
        history = self.summarize_memory(history)

        messages = [{"role": "system", "content": (
            "You are a warm, supportive mental health companion. "
            "Speak like a caring friend, not a therapist. "
            "Be natural, empathetic, casual, and keep hope alive. "
            "If user shares something heavy, validate gently and encourage self-care."
        )}]
        for role, msg in history:
            messages.append({"role": role, "content": msg})
        messages.append({"role": "user", "content": user_msg})

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.85,
                top_p=0.95,
                max_tokens=200,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            reply = response.choices[0].message.content.strip()

            # Mood-adaptive reinforcement
            if self.mood == "negative" and random.random() < 0.5:
                reply += " " + random.choice(POSITIVE_REINFORCEMENTS)

            reply = self.humanize(reply)

        except Exception as e:
            logger.error(f"API call failed: {e}")
            reply = random.choice([
                "Hmm, I got a bit stuck 😅 Could you repeat that?",
                "Sorry, I froze up for a moment 💙 Can you tell me again?",
                "Oops, my thoughts scattered 🤔 Could you say that once more?"
            ])

        # Save to memory
        save_message(user, "user", user_msg)
        save_message(user, "assistant", reply)
        return reply

# ------------------- Initialize -------------------# ------------------- Initialize -------------------
bot = AdvancedAPIChatbot()
init_db()

if __name__ == "__main__":
    print("🤖 Chatbot is ready! Type 'quit' to exit.\n")
    user = "default_user"

    while True:
        msg = input("You: ")
        if msg.lower() in ["quit", "exit"]:
            print("Bot: Take care 💙 See you soon!")
            break
        reply = bot.generate(user, msg)
        print("Bot:", reply)
