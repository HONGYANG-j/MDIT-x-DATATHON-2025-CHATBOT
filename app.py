# app.py - Advanced Mental Health AI with Anti-Repetition
import random
import time
import logging
import re
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.secret_key = "super-simple-secret"


# GPU/Device detection
def setup_device():
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU detected: {gpu_name}")
            logger.info(f"✅ CUDA version: {torch.version.cuda}")
            logger.info(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
            return device, 0
        else:
            logger.warning("⚠️ Using CPU - responses may be slower")
            return torch.device("cpu"), -1
    except Exception as e:
        logger.error(f"❌ Device detection failed: {e}")
        return torch.device("cpu"), -1


device, device_id = setup_device()

# Load advanced model
chatbot_model = None
tokenizer = None

try:
    logger.info("🚀 Loading advanced mental health model...")

    # Use a more capable model
    model_name = "microsoft/DialoGPT-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move to appropriate device with optimization
    model = model.to(device)

    if device_id == 0:  # GPU
        model = model.half()
    else:
        model = model.float()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Create optimized pipeline
    chatbot_model = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
        torch_dtype=torch.float16 if device_id == 0 else torch.float32
    )

    logger.info(f"✅ Advanced model loaded successfully on {'GPU' if device_id == 0 else 'CPU'}")

except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    chatbot_model = None


class AdvancedMentalHealthChatbot:
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.user_profile: Dict = {
            'name': None,
            'mood_trend': [],
            'topics_discussed': [],
            'communication_style': 'reflective'
        }
        self.response_memory: List[str] = []
        self.last_phrases: List[str] = []
        self.conversation_depth = 0

    def analyze_emotional_content(self, text: str) -> Dict:
        """Advanced emotional analysis with sentiment detection"""
        text_lower = text.lower()

        emotional_categories = {
            'positive': ['happy', 'good', 'great', 'better', 'improved', 'hopeful', 'calm', 'peaceful'],
            'negative': ['sad', 'bad', 'awful', 'terrible', 'anxious', 'stressed', 'overwhelmed', 'tired'],
            'crisis': ['suicide', 'self-harm', 'end life', 'want to die', 'cant continue', 'hopeless'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'usual', 'same']
        }

        emotional_score = {
            'primary_emotion': 'neutral',
            'intensity': 0.5,
            'word_count': len(text.split()),
            'has_question': '?' in text,
            'emotional_words': []
        }

        # Analyze emotional content
        for category, words in emotional_categories.items():
            for word in words:
                if word in text_lower:
                    emotional_score['emotional_words'].append(word)
                    if category != 'neutral':
                        emotional_score['intensity'] += 0.1

        # Determine primary emotion
        emotion_counts = {cat: 0 for cat in emotional_categories.keys()}
        for word in emotional_score['emotional_words']:
            for category, words in emotional_categories.items():
                if word in words:
                    emotion_counts[category] += 1

        if emotion_counts['crisis'] > 0:
            emotional_score['primary_emotion'] = 'crisis'
        elif emotion_counts['negative'] > emotion_counts['positive']:
            emotional_score['primary_emotion'] = 'negative'
        elif emotion_counts['positive'] > emotion_counts['negative']:
            emotional_score['primary_emotion'] = 'positive'

        emotional_score['intensity'] = min(1.0, max(0.1, emotional_score['intensity']))

        return emotional_score

    def detect_repetition_patterns(self, text: str) -> bool:
        """Advanced repetition detection"""
        text_lower = text.lower()

        # Check for duplicate phrases
        words = text_lower.split()
        if len(words) < 5:
            return False

        # Check for repeated 3-gram patterns
        for i in range(len(words) - 2):
            trigram = ' '.join(words[i:i + 3])
            if text_lower.count(trigram) > 1:
                return True

        # Check for known problematic patterns
        problematic_patterns = [
            r'\b(i want to acknowledge)\b.*\1',
            r'\b(it makes complete sense)\b.*\1',
            r'\b(thank you for sharing)\b.*\1',
            r'(\b\w+\b).*\1.*\1'  # same word repeated 3 times
        ]

        for pattern in problematic_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    def generate_advanced_prompt(self, user_msg: str) -> str:
        """Create sophisticated, non-repetitive prompt"""
        emotional_analysis = self.analyze_emotional_content(user_msg)

        # Build diverse context
        context_lines = []
        for msg in self.conversation_history[-3:]:
            role = "Patient" if msg["role"] == "user" else "Therapist"
            context_lines.append(f"{role}: {msg['msg']}")

        context = "\n".join(context_lines)

        # Diverse prompt templates to prevent repetition
        prompt_templates = [
            """As an expert AI mental health therapist, provide a thoughtful response that demonstrates:

1. Deep emotional validation without repetition
2. Unique insights tailored to this specific situation
3. Evidence-based therapeutic perspective
4. Open-ended question to encourage exploration
5. Warm, professional tone with varied phrasing

Recent conversation:
{context}

Patient's emotional state: {emotion} (intensity: {intensity})
Patient: {message}

Therapist: """,

            """Mental Health AI Response Guidelines:
- Avoid repetitive phrases from previous responses
- Provide fresh, unique perspective each time
- Use varied sentence structures and vocabulary
- Incorporate different therapeutic approaches
- Maintain genuine, non-repetitive empathy

Conversation context:
{context}

Current patient message: {message}
Detected emotion: {emotion}

Response: """,

            """Therapeutic AI Response (Avoid repetition):
Create a response that doesn't reuse phrases from: {last_phrases}

Key elements:
1. Original validation statement
2. Unique psychological insight
3. Practical suggestion
4. Thought-provoking question
5. Varied vocabulary and structure

Patient: {message}
Emotion: {emotion}

Therapist: """
        ]

        prompt_template = random.choice(prompt_templates)

        prompt = prompt_template.format(
            context=context,
            emotion=emotional_analysis['primary_emotion'],
            intensity=emotional_analysis['intensity'],
            message=user_msg,
            last_phrases=", ".join(self.last_phrases[-3:]) if self.last_phrases else "none"
        )

        return prompt

    def get_ai_response(self, user_msg: str) -> str:
        """Get high-quality, non-repetitive AI response"""
        if not chatbot_model:
            return self.generate_diverse_fallback(user_msg)

        try:
            start_time = time.time()

            prompt = self.generate_advanced_prompt(user_msg)

            # Generate with anti-repetition parameters
            output = chatbot_model(
                prompt,
                max_new_tokens=100,
                num_return_sequences=3,  # Generate multiple options
                temperature=0.85,  # Higher temperature for diversity
                top_p=0.92,
                top_k=60,
                repetition_penalty=1.4,  # Strong anti-repetition
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                truncation=True,
                num_beams=4,
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                early_stopping=True
            )

            # Select the best response from multiple options
            best_response = self.select_best_response([opt["generated_text"] for opt in output], user_msg)

            generation_time = time.time() - start_time
            logger.info(f"🧠 Advanced response generated in {generation_time:.2f}s")

            return best_response

        except Exception as e:
            logger.error(f"Advanced AI generation error: {e}")
            return self.generate_diverse_fallback(user_msg)

    def select_best_response(self, responses: List[str], user_msg: str) -> str:
        """Select the best response from multiple options"""
        scored_responses = []

        for response in responses:
            # Clean response
            clean_response = self.clean_response(response)
            if not clean_response or len(clean_response.split()) < 10:
                continue

            # Score based on quality metrics
            score = self.score_response_quality(clean_response, user_msg)
            scored_responses.append((score, clean_response))

        if not scored_responses:
            return self.generate_diverse_fallback(user_msg)

        # Select highest scoring response
        scored_responses.sort(key=lambda x: x[0], reverse=True)
        best_response = scored_responses[0][1]

        # Store phrases to avoid repetition
        self.update_phrase_memory(best_response)

        return best_response

    def score_response_quality(self, response: str, user_msg: str) -> float:
        """Score response based on multiple quality metrics"""
        score = 0.0

        # Length score (prefer 20-40 words)
        word_count = len(response.split())
        if 20 <= word_count <= 40:
            score += 2.0
        elif 15 <= word_count <= 50:
            score += 1.0

        # Diversity score (avoid repetition)
        if not self.detect_repetition_patterns(response):
            score += 3.0

        # Emotional alignment
        user_emotion = self.analyze_emotional_content(user_msg)['primary_emotion']
        response_emotion = self.analyze_emotional_content(response)['primary_emotion']
        if response_emotion in ['positive', 'neutral'] and user_emotion in ['negative', 'crisis']:
            score += 2.0  # Positive but appropriate response to negative input

        # Question presence (good for conversation)
        if '?' in response:
            score += 1.0

        # Avoid recent phrases
        recent_phrase_penalty = 0
        for phrase in self.last_phrases:
            if phrase.lower() in response.lower():
                recent_phrase_penalty += 1.0
        score -= recent_phrase_penalty

        return max(0, score)

    def clean_response(self, response: str) -> str:
        """Advanced response cleaning"""
        if not response:
            return ""

        # Remove prompt fragments
        response = re.sub(r'.*Therapist:\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'Patient:.*', '', response, flags=re.IGNORECASE)

        # Remove common artifacts
        artifacts = [
            r'\[.*?\]', r'\(.*?\)', r'<.*?>',
            r'\b(um|uh|hmm|well|so|like|actually|basically)\b',
            r'\s+\.', r'\.\.+', r',,+,'
        ]

        for pattern in artifacts:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)

        # Ensure proper formatting
        response = response.strip()
        if response:
            response = response[0].upper() + response[1:]
            if not response.endswith(('.', '!', '?')):
                response += '.'

        return response

    def update_phrase_memory(self, response: str):
        """Store phrases to avoid repetition"""
        # Extract key phrases (3-4 word n-grams)
        words = response.lower().split()
        phrases = []

        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i + 3])
            if len(phrase) > 10 and not any(word in phrase for word in ['the', 'and', 'but', 'for']):
                phrases.append(phrase)

        # Add to memory (keep last 10 phrases)
        self.last_phrases.extend(phrases)
        self.last_phrases = self.last_phrases[-10:]

    def generate_diverse_fallback(self, user_msg: str) -> str:
        """Highly diverse fallback responses"""
        emotional_state = self.analyze_emotional_content(user_msg)

        if emotional_state['primary_emotion'] == 'crisis':
            return self._get_crisis_response()

        # Extensive diverse response bank
        response_banks = {
            'negative': [
                "I can sense the weight behind your words, and I want you to know that being able to articulate these feelings is itself a significant step forward. What aspect of this has been most present for you recently?",
                "Thank you for trusting me with this. It sounds like you're navigating some complex emotional terrain. I'm wondering what small moment of relief might look like for you right now?",
                "I hear the depth of what you're sharing. These moments, while challenging, often contain important insights about what we need. What's one thing that usually helps you feel even slightly more grounded?"
            ],
            'positive': [
                "It's heartening to hear that you're experiencing some positive momentum! What do you attribute this shift to, and how might you build upon it?",
                "I'm genuinely pleased to hear this. These brighter moments can be powerful teachers about what supports our wellbeing. What have you discovered about yourself during this time?",
                "That's wonderful to hear! Celebrating these positive experiences helps reinforce them. What's been different in your approach or perspective lately?"
            ],
            'neutral': [
                "I appreciate you sharing this. Sometimes these neutral spaces allow for important reflection. What's been occupying your thoughts beneath the surface?",
                "Thank you for checking in. Even in quieter moments, there's often valuable self-awareness developing. What would you like to explore together today?",
                "I'm here with you in this space. These calmer periods can be opportunities for understanding ourselves more deeply. What's been on your mind lately?"
            ]
        }

        # Ensure no repetition from recent responses
        available_responses = response_banks[emotional_state['primary_emotion']]
        for response in available_responses:
            if not any(phrase in response.lower() for phrase in self.last_phrases):
                return response

        return random.choice(available_responses)

    def _get_crisis_response(self) -> str:
        """Non-repetitive crisis response"""
        crisis_responses = [
            """💙 I hear the profound depth of what you're sharing, and I want you to know that your presence matters immensely. 

What you're experiencing right now represents an incredible weight, but it's crucial to remember that these intense feelings, while overwhelming, can shift with proper support. Many have walked similar paths and found their way through with compassionate help.

**Immediate Support Available:**
📞 **Befrienders KL**: +603-7627 2929 (24/7, completely confidential)
📞 **Talian Kasih**: 15999 (24/7, government mental health support)
📞 **Emergency Services**: 999 (immediate danger)

You deserve professional support from trained individuals who can provide appropriate care. Would you consider reaching out to one of these resources? I'm here with you.""",

            """💙 I want to acknowledge the tremendous courage it takes to share something this profound. Your life has inherent value and meaning.

What you're describing represents significant emotional pain, but it's important to know that these feelings, while overwhelming, are not permanent. Many people have experienced similar depths and found pathways forward with proper support.

**Available Right Now:**
📞 **Befrienders KL**: +603-7627 2929 (multilingual, 24/7)
📞 **Talian Kasih**: 15999 (government mental health line)
📞 **Emergency Services**: 999 (immediate help)

Professional support can make a profound difference. Would you be open to contacting one of these resources? I'm here with you through this."""
        ]

        return random.choice(crisis_responses)

    def get_response(self, user_msg: str) -> str:
        """Main method with advanced anti-repetition"""
        start_time = time.time()

        # Crisis detection
        emotional_state = self.analyze_emotional_content(user_msg)
        if emotional_state['primary_emotion'] == 'crisis':
            return self._get_crisis_response()

        # Update conversation history
        self.conversation_history.append({"role": "user", "msg": user_msg})
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]

        # Get response with quality assurance
        response = self.get_ai_response(user_msg)
        response = self.final_quality_check(response, user_msg)

        # Update history and memory
        self.conversation_history.append({"role": "assistant", "msg": response})
        self.conversation_depth += 1

        total_time = time.time() - start_time
        logger.info(f"🎯 Total processing time: {total_time:.2f}s")

        return response

    def final_quality_check(self, response: str, user_msg: str) -> str:
        """Final anti-repetition and quality validation"""
        if not response or len(response.split()) < 12:
            return self.generate_diverse_fallback(user_msg)

        # Check for repetition
        if self.detect_repetition_patterns(response):
            logger.warning("🔄 Repetition detected - generating alternative")
            return self.generate_diverse_fallback(user_msg)

        # Ensure professional quality
        if any(unprofessional in response.lower() for unprofessional in ['stupid', 'dumb', 'hate', 'suck', 'whatever']):
            return self.generate_diverse_fallback(user_msg)

        return response


# Initialize advanced chatbot
bot = AdvancedMentalHealthChatbot()


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.form.get("msg", "").strip()
    if not user_msg:
        return jsonify({"reply": "I'm here to listen and support you 💙 What would you like to share today?"})

    try:
        bot_reply = bot.get_response(user_msg)
        return jsonify({"reply": bot_reply})
    except Exception as e:
        logger.error(f"Advanced chat error: {e}")
        return jsonify(
            {"reply": "I'm here with you 💙 Let's continue our conversation - what's been on your mind lately?"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)