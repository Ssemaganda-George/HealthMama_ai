from flask import Flask, request, jsonify, render_template, session
import openai
import numpy as np
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='Templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY', b'\xdb\x908\x9bsKD\x1c\x91\x8a\xd84\x01\xcb\xa5]\x8b\xa9n\x10\xd7\x1e\x11g')

# Load cleaned data
def load_cleaned_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
    except FileNotFoundError:
        print(f"Warning: Data file not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

# Simplified context retrieval using keyword matching instead of embeddings
def retrieve_context_simple(query, data, top_n=5):
    """Simple keyword-based context retrieval"""
    query_words = query.lower().split()
    scored_data = []
    
    for idx, text in enumerate(data):
        score = sum(1 for word in query_words if word in text.lower())
        if score > 0:
            scored_data.append((score, idx, text))
    
    # Sort by score and return top results
    scored_data.sort(reverse=True)
    return [text for _, _, text in scored_data[:top_n]]

# Initialize conversation history
conversation_history_dict = {
    'diabetes': [],
    'preeclampsia': []
}

# Load data once at startup
diabetes_data = []
preeclampsia_data = []

try:
    diabetes_data = load_cleaned_data('data_diabetes/cleaned_diabetes.txt')
    print(f"Loaded {len(diabetes_data)} diabetes data entries")
except Exception as e:
    print(f"Warning: Could not load diabetes data: {e}")

try:
    preeclampsia_data = load_cleaned_data('data_preelampsia/cleaned_preecampsia.txt')
    print(f"Loaded {len(preeclampsia_data)} preeclampsia data entries")
except Exception as e:
    print(f"Warning: Could not load preeclampsia data: {e}")

# Generate response with OpenAI API
def generate_response_with_openai(conversation_history, model='diabetes'):
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "Error: OpenAI API key not configured"
        
        # Clean the API key of any extra whitespace or characters
        api_key = api_key.strip().replace('\t', '').replace('\n', '').replace('\r', '')
        
        openai.api_key = api_key
        
        if model == 'preeclampsia':
            system_prompt = (
                "You are a helpful assistant specialized in health information, with a focus on preeclampsia and maternal health. "
                "Always respond in the same language as the user's question. Detect the user's language and reply in that language. "
                "Provide accurate, concise, and informative responses based on the given context. "
                "If the question is not related to health or preeclampsia, politely inform the user that you can only provide information on health and preeclampsia."
            )
        else:  # diabetes
            system_prompt = (
                "You are a helpful assistant specialized in health information, with a focus on diabetes and blood sugar management. "
                "Always respond in the same language as the user's question. Detect the user's language and reply in that language. "
                "Provide accurate, concise, and informative responses based on the given context. "
                "If the question is not related to health or diabetes, politely inform the user that you can only provide information on health and diabetes."
            )

        messages = [{"role": "system", "content": system_prompt}] + conversation_history

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
        
    except openai.error.AuthenticationError:
        return "Error: Invalid OpenAI API key. Please check your configuration."
    except openai.error.RateLimitError:
        return "Error: OpenAI API rate limit exceeded. Please try again later."
    except openai.error.APIError as e:
        return f"Error: OpenAI API error - {str(e)}"
    except Exception as e:
        print(f"OpenAI error: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"HealthMama AI is running, but template error: {str(e)}", 200

@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'message': 'HealthMama AI is running',
        'data_loaded': {
            'diabetes': len(diabetes_data),
            'preeclampsia': len(preeclampsia_data)
        }
    }), 200

@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """Test endpoint to debug issues"""
    api_key = os.getenv('OPENAI_API_KEY')
    return jsonify({
        'method': request.method,
        'endpoint': 'test',
        'status': 'working',
        'openai_key_configured': bool(api_key),
        'api_key_length': len(api_key) if api_key else 0,
        'api_key_starts_with': api_key[:20] + '...' if api_key else 'None',
        'data_status': {
            'diabetes_loaded': len(diabetes_data) > 0,
            'preeclampsia_loaded': len(preeclampsia_data) > 0
        }
    }), 200

@app.route('/ask', methods=['POST'])
def ask():
    """Legacy endpoint that matches the frontend - converts FormData to JSON format"""
    try:
        # Handle FormData from frontend
        if request.form:
            user_message = request.form.get('query', '').strip()
            model = request.form.get('model', 'diabetes')
        elif request.is_json:
            data = request.get_json()
            user_message = data.get('message', '').strip()
            model = data.get('model', 'diabetes')
        else:
            return jsonify({'error': 'No data received', 'response': 'Please provide valid data.'}), 400
            
        if not user_message:
            return jsonify({
                'response': 'Please provide a message.', 
                'conversation_history': conversation_history_dict.get(model, []), 
                'audio_url': None
            })

        # Get relevant context using simple keyword matching
        if model == 'preeclampsia' and preeclampsia_data:
            context = retrieve_context_simple(user_message, preeclampsia_data)
        elif model == 'diabetes' and diabetes_data:
            context = retrieve_context_simple(user_message, diabetes_data)
        else:
            context = []

        # Add context to the conversation
        context_text = "\n".join(context[:3]) if context else "No specific context available."
        
        # Add user message to conversation history
        conversation_history_dict[model].append({"role": "user", "content": user_message})
        
        # Add context as system message
        if context:
            conversation_history_dict[model].append({"role": "system", "content": f"Relevant context: {context_text}"})

        # Keep conversation history manageable
        if len(conversation_history_dict[model]) > 10:
            conversation_history_dict[model] = conversation_history_dict[model][-8:]

        response = generate_response_with_openai(conversation_history_dict[model], model)
        
        # Add AI response to conversation history
        conversation_history_dict[model].append({"role": "assistant", "content": response})

        return jsonify({
            'response': response, 
            'conversation_history': conversation_history_dict[model], 
            'audio_url': None
        })
        
    except Exception as e:
        print(f"Error in ask endpoint: {str(e)}")
        return jsonify({
            'error': f'Ask error: {str(e)}',
            'response': 'Sorry, I encountered an error processing your request.',
            'conversation_history': conversation_history_dict.get(model if 'model' in locals() else 'diabetes', []),
            'audio_url': None
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        start_time = time.time()
        
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON', 'response': 'Please send JSON data.'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received', 'response': 'Please provide valid JSON.'}), 400
            
        user_message = data.get('message', '').strip()
        model = data.get('model', 'diabetes')  
        
        if not user_message:
            return jsonify({
                'response': 'Please provide a message.', 
                'conversation_history': conversation_history_dict[model], 
                'audio_url': None
            })

        # Get relevant context using simple keyword matching
        if model == 'preeclampsia' and preeclampsia_data:
            context = retrieve_context_simple(user_message, preeclampsia_data)
        elif model == 'diabetes' and diabetes_data:
            context = retrieve_context_simple(user_message, diabetes_data)
        else:
            context = []

        # Add context to the conversation
        context_text = "\n".join(context[:3]) if context else "No specific context available."
        
        # Add user message to conversation history
        conversation_history_dict[model].append({"role": "user", "content": user_message})
        
        # Add context as system message
        if context:
            conversation_history_dict[model].append({"role": "system", "content": f"Relevant context: {context_text}"})

        # Keep conversation history manageable
        if len(conversation_history_dict[model]) > 10:
            conversation_history_dict[model] = conversation_history_dict[model][-8:]

        response = generate_response_with_openai(conversation_history_dict[model], model)
        
        # Add AI response to conversation history
        conversation_history_dict[model].append({"role": "assistant", "content": response})

        response_time = time.time()
        print(f"Time to get response: {response_time - start_time} seconds")

        return jsonify({
            'response': response, 
            'conversation_history': conversation_history_dict[model], 
            'audio_url': None
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': f'Chat error: {str(e)}',
            'response': 'Sorry, I encountered an error processing your request.',
            'conversation_history': conversation_history_dict.get(request.get_json().get('model', 'diabetes') if request.is_json else 'diabetes', []),
            'audio_url': None
        }), 500

if __name__ == '__main__':
    # Use environment variable for port (for deployment platforms like Railway)
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)