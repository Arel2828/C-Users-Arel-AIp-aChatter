from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import json
from flask_cors import CORS  # To handle CORS issues

# Load model and tokenizer
model_name = "rinna/japanese-gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load teaching data from JSON file
def load_teaching_data(filename='teaching_data.json'):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            teaching_data = json.load(file)
        return teaching_data
    except FileNotFoundError:
        return {}  # Return empty dictionary if the file is not found

teaching_data = load_teaching_data()

@app.route('/')
def home():
    return render_template('index.html')

# Detect the question type for targeted responses
def detect_question_type(user_input):
    if user_input.startswith("何") or user_input.startswith("なに"):  # "What"
        return "事実を簡潔に述べてください: "  # "Please state the facts concisely: "
    elif user_input.startswith("どう") or user_input.startswith("どのように"):  # "How"
        return "手順を簡単に説明してください: "  # "Please briefly explain the steps: "
    elif user_input.startswith("なぜ") or user_input.startswith("何故"):  # "Why"
        return "理由を説明してください: "  # "Please explain the reason: "
    else:
        return "以下の質問に対して、適切で具体的な回答を提供してください: "  # General prompt

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '').strip()
        if not user_input:
            return jsonify({"response": "Please provide a valid message."})
        
        # Check for specific input and provide a predefined response
        if user_input.lower() == 'quit':
            return jsonify({"response": "Exiting chat. Goodbye!"})
        elif user_input.lower() == "what is your name" or user_input == "あなたの名前は何ですか":
            return jsonify({"response": "Hi, my name is 日本の家庭教師, it means Japanese tutor。"})

        # Detect question type and construct tailored prompt
        prompt = detect_question_type(user_input)
        engineered_input = prompt + user_input

        # Add teaching data context if available
        if user_input in teaching_data:
            teaching_context = teaching_data[user_input]
            engineered_input = teaching_context + "\n" + engineered_input  # Append teaching context

        # Encode input and generate response
        inputs = tokenizer.encode(engineered_input, return_tensors='pt')
        chat_history_ids = model.generate(
            inputs,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=10,
            top_p=0.8,
            temperature=0.5,
            do_sample=True
        )

        # Decode and post-process the response
        bot_response = tokenizer.decode(chat_history_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        bot_response = post_process_response(bot_response)
        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"Error processing the request: {e}")
        return jsonify({"response": "Sorry, I encountered an error. Please try again later."})

# Post-process response to keep it concise
def post_process_response(response):
    response = response.split("。")[0] + "。"  # Simplify by taking only the first sentence
    return response.strip()

# Optional: Fine-tune model with Hugging Face datasets
def fine_tune_model():
    # Load datasets (e.g., JESC, Wiki40B)
    dataset = load_dataset("wiki40b", "ja")  # Replace with other datasets like "jesc" or "jparacrawl" as needed

    # Tokenize dataset
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Fine-tuning configuration
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    # Fine-tune the model
    trainer.train()

# Uncomment below if you want to fine-tune before starting the app
# fine_tune_model()

if __name__ == "__main__":
    app.run(debug=True)
