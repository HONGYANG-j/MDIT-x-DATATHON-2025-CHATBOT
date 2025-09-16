from flask import Flask, render_template, request, jsonify, session
from chatbot import bot   # Import your chatbot instance
import os


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "super-secret-key")

@app.route("/")
def home():
    if "user" not in session:
        session["user"] = os.urandom(8).hex()
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("msg")
    user = session["user"]
    reply = bot.generate(user, user_msg)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
