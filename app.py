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
    """Simple keyword-based context retrieval with better matching for different languages"""
    query_words = query.lower().split()
    scored_data = []
    
    # Multilingual health term mappings
    multilingual_terms = {
        # Luganda terms
        'ssukali': ['diabetes', 'sugar', 'blood sugar'],
        'musujja': ['diabetes', 'sugar'],
        'omusujja': ['diabetes', 'sugar'],
        'omusayi': ['blood', 'pressure'],
        'omutwe': ['head', 'headache'],
        'olumbe': ['disease', 'illness'],
        'obulwadde': ['disease', 'illness', 'condition'],
        'okujjukira': ['remember', 'memory'],
        'endya': ['food', 'eat', 'diet'],
        'emmere': ['food', 'meal', 'diet'],
        'amabeere': ['milk', 'dairy'],
        'ebijjanjalo': ['vegetables'],
        'ebibala': ['fruits'],
        'amazzi': ['water'],
        'okutambula': ['walk', 'exercise'],
        'okukola': ['work', 'exercise'],
        
        # Swahili terms
        'sukari': ['diabetes', 'sugar', 'blood sugar'],
        'ugonjwa': ['disease', 'illness', 'condition'],
        'damu': ['blood', 'pressure'],
        'kichwa': ['head', 'headache'],
        'chakula': ['food', 'meal', 'diet'],
        'maziwa': ['milk', 'dairy'],
        'mboga': ['vegetables'],
        'matunda': ['fruits'],
        'maji': ['water'],
        'kutembea': ['walk', 'exercise'],
        'kufanya': ['work', 'exercise'],
        'mama': ['mother', 'pregnancy'],
        'mtoto': ['child', 'baby'],
        
        # French terms
        'diabète': ['diabetes', 'sugar', 'blood sugar'],
        'sucre': ['sugar', 'blood sugar'],
        'maladie': ['disease', 'illness', 'condition'],
        'sang': ['blood', 'pressure'],
        'tête': ['head', 'headache'],
        'nourriture': ['food', 'meal', 'diet'],
        'lait': ['milk', 'dairy'],
        'légumes': ['vegetables'],
        'fruits': ['fruits'],
        'eau': ['water'],
        'marcher': ['walk', 'exercise'],
        'mère': ['mother', 'pregnancy'],
        'enfant': ['child', 'baby'],
        
        # Arabic terms
        'سكري': ['diabetes', 'sugar', 'blood sugar'],
        'سكر': ['sugar', 'blood sugar'],
        'مرض': ['disease', 'illness', 'condition'],
        'دم': ['blood', 'pressure'],
        'رأس': ['head', 'headache'],
        'طعام': ['food', 'meal', 'diet'],
        'حليب': ['milk', 'dairy'],
        'خضار': ['vegetables'],
        'فواكه': ['fruits'],
        'ماء': ['water'],
        'مشي': ['walk', 'exercise'],
        'أم': ['mother', 'pregnancy'],
        'طفل': ['child', 'baby'],
        
        # Spanish terms
        'azúcar': ['diabetes', 'sugar', 'blood sugar'],
        'enfermedad': ['disease', 'illness', 'condition'],
        'sangre': ['blood', 'pressure'],
        'cabeza': ['head', 'headache'],
        'comida': ['food', 'meal', 'diet'],
        'leche': ['milk', 'dairy'],
        'verduras': ['vegetables'],
        'frutas': ['fruits'],
        'agua': ['water'],
        'caminar': ['walk', 'exercise'],
        'madre': ['mother', 'pregnancy'],
        'niño': ['child', 'baby']
    }
    
    # Expand query with related terms from multiple languages
    expanded_query = set(query_words)
    for word in query_words:
        if word in multilingual_terms:
            expanded_query.update(multilingual_terms[word])
    
    expanded_query = list(expanded_query)
    
    for idx, text in enumerate(data):
        text_lower = text.lower()
        score = sum(1 for word in expanded_query if word in text_lower)
        if score > 0:
            scored_data.append((score, idx, text))
    
    # Sort by score and return top results
    scored_data.sort(reverse=True)
    return [text for _, _, text in scored_data[:top_n]]

def detect_language_and_add_cultural_context(user_message, conversation_history, model):
    """Detect various languages and add specific cultural context for better responses"""
    
    # Language detection patterns
    language_patterns = {
        'luganda': {
            'indicators': ['nga', 'bwe', 'ku', 'mu', 'nti', 'kiki', 'ani', 'wa', 'gw', 'ly', 'gy', 'ssukali', 'omusujja', 'obulwadde', 'emmere', 'amazzi', 'otya', 'oli', 'nkwagala'],
            'cultural_context': "The user is asking in Luganda. Please respond in clear, detailed Luganda using simple terms. Include practical advice relevant to Ugandan context. Use common Luganda health vocabulary and explain medical terms clearly. Give specific examples with local foods like matooke, posho, beans, cassava when discussing diet. Consider traditional Ugandan healthcare practices."
        },
        'swahili': {
            'indicators': ['habari', 'hujambo', 'asante', 'karibu', 'chakula', 'maji', 'dawa', 'ugonjwa', 'afya', 'daktari', 'hospitali', 'sukari', 'damu', 'mzazi', 'mama', 'mtoto'],
            'cultural_context': "The user is asking in Swahili. Please respond in clear, detailed Swahili using simple terms. Include practical advice relevant to East African context. Use common Swahili health vocabulary and explain medical terms clearly. Give examples with local foods like ugali, rice, beans, vegetables commonly found in East Africa."
        },
        'french': {
            'indicators': ['bonjour', 'salut', 'merci', 'comment', 'allez', 'vous', 'santé', 'maladie', 'médecin', 'hôpital', 'diabète', 'sucre', 'sang', 'grossesse', 'mère', 'enfant', 'nourriture'],
            'cultural_context': "The user is asking in French. Please respond in clear, detailed French using simple terms. Include practical advice relevant to French-speaking African context. Use common French health vocabulary and explain medical terms clearly. Consider local dietary habits and healthcare practices in French-speaking regions."
        },
        'arabic': {
            'indicators': ['السلام', 'أهلا', 'مرحبا', 'شكرا', 'صحة', 'مرض', 'طبيب', 'مستشفى', 'سكري', 'دم', 'حمل', 'أم', 'طفل', 'طعام', 'ماء'],
            'cultural_context': "The user is asking in Arabic. Please respond in clear, detailed Arabic using simple terms. Include practical advice relevant to Arabic-speaking context. Use common Arabic health vocabulary and explain medical terms clearly. Consider cultural dietary practices and healthcare traditions in Arabic-speaking communities."
        },
        'spanish': {
            'indicators': ['hola', 'gracias', 'cómo', 'está', 'salud', 'enfermedad', 'doctor', 'hospital', 'diabetes', 'azúcar', 'sangre', 'embarazo', 'madre', 'niño', 'comida'],
            'cultural_context': "The user is asking in Spanish. Please respond in clear, detailed Spanish using simple terms. Include practical advice relevant to Spanish-speaking context. Use common Spanish health vocabulary and explain medical terms clearly. Consider local dietary habits and healthcare practices."
        },
        'portuguese': {
            'indicators': ['olá', 'obrigado', 'como', 'está', 'saúde', 'doença', 'médico', 'hospital', 'diabetes', 'açúcar', 'sangue', 'gravidez', 'mãe', 'criança', 'comida'],
            'cultural_context': "The user is asking in Portuguese. Please respond in clear, detailed Portuguese using simple terms. Include practical advice relevant to Portuguese-speaking African context. Use common Portuguese health vocabulary and explain medical terms clearly."
        },
        'amharic': {
            'indicators': ['ሰላም', 'አመሰግናለሁ', 'እንዴት', 'ጤና', 'በሽታ', 'ሐኪም', 'ሆስፒታል', 'ስኳር', 'ደም', 'እርግዝና', 'እናት', 'ልጅ', 'ምግብ'],
            'cultural_context': "The user is asking in Amharic. Please respond in clear, detailed Amharic using simple terms. Include practical advice relevant to Ethiopian context. Use common Amharic health vocabulary and explain medical terms clearly. Consider traditional Ethiopian dietary practices and healthcare."
        }
    }
    
    # Detect language
    detected_language = None
    user_message_lower = user_message.lower()
    
    for language, patterns in language_patterns.items():
        if any(indicator in user_message_lower for indicator in patterns['indicators']):
            detected_language = language
            break
    
    # Add cultural context if language detected
    if detected_language:
        conversation_history.append({
            "role": "system", 
            "content": language_patterns[detected_language]['cultural_context']
        })
    
    return conversation_history

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
                "CRITICAL: Always respond in the EXACT same language as the user's question. Automatically detect the user's language and reply in that language. "
                "Support multiple languages including English, Luganda, Swahili, French, Arabic, Spanish, Portuguese, Amharic, and others. "
                "When responding in local languages, provide detailed, culturally appropriate responses using simple, clear terms. "
                "Include practical advice that is relevant to the local healthcare context and cultural practices. "
                "Use common local health terms and explain medical concepts in ways that are easily understood. "
                "For African contexts, consider traditional practices, local foods, and healthcare accessibility. "
                "Provide accurate, concise, and informative responses based on the given context. "
                "If the question is not related to health or preeclampsia, politely inform the user that you can only provide information on health and preeclampsia."
            )
        else:  # diabetes
            system_prompt = (
                "You are a helpful assistant specialized in health information, with a focus on diabetes and blood sugar management. "
                "CRITICAL: Always respond in the EXACT same language as the user's question. Automatically detect the user's language and reply in that language. "
                "Support multiple languages including English, Luganda, Swahili, French, Arabic, Spanish, Portuguese, Amharic, and others. "
                "When responding in local languages, provide detailed, culturally appropriate responses using simple, clear terms. "
                "Include practical advice about diet, exercise, and diabetes management that is relevant to local lifestyle and available foods. "
                "Use common local health terms and explain medical concepts in ways that are easily understood. "
                "For African contexts, mention local foods like matooke, posho, ugali, cassava, beans, and their effects on blood sugar when relevant. "
                "Consider traditional practices, cultural dietary habits, and healthcare accessibility in your responses. "
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
        
        # Add language-specific cultural context  
        conversation_history_dict[model] = detect_language_and_add_cultural_context(
            user_message, conversation_history_dict[model], model
        )
        
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
        
        # Add language-specific cultural context
        conversation_history_dict[model] = detect_language_and_add_cultural_context(
            user_message, conversation_history_dict[model], model
        )
        
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