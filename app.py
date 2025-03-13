from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load chatbot model
model_dir = "C:\\Users\\puran\\AppData\\Local\\Programs\\Python\\Python311\\srec_trained_gpt2_chatbot"


tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    query = f"Q: {user_input}\\nA:"
    response = chatbot(query, max_length=100)
    return jsonify({"response": response[0]["generated_text"]})

if __name__ == "__main__":
    app.run(debug=True)
