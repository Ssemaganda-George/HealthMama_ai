from flask import Flask, request, jsonify, render_template, session
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', b'\xdb\x908\x9bsKD\x1c\x91\x8a\xd84\x01\xcb\xa5]\x8b\xa9n\x10\xd7\x1e\x11g')  # Use env var or fallback

# Load cleaned data
def load_cleaned_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Load embeddings from file
def load_embeddings(file_path):
    return np.load(file_path)

# # Retrieve context using cosine similarity 
# def retrieve_context(query, index, data, top_n=5):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     query_embedding = model.encode([query], convert_to_tensor=True)
#     similarities = cosine_similarity(query_embedding, index)[0]
#     top_indices = similarities.argsort()[-top_n:][::-1]
#     return [data[idx] for idx in top_indices]

def retrieve_context(query, index, data, top_n=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_tensor=True)
    
    # Move the tensor to CPU if it's on MPS or any other GPU
    query_embedding = query_embedding.cpu().numpy()  # Convert tensor to NumPy array
    
    similarities = cosine_similarity(query_embedding, index)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [data[idx] for idx in top_indices]



# Generate response with OpenAI API

def generate_response_with_openai(conversation_history, model='diabetes'):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if model == 'preeclampsia':
        system_prompt = (
            "You are a helpful assistant specialized in health information, with a focus on preeclampsia and maternal health. "
            "Always respond in the same language as the user's question. Detect the user's language and reply in that language. "
            "Provide accurate, concise, and informative responses based on the given context. "
            "If the question is not related to health or preeclampsia, politely inform the user that you can only provide information on health and preeclampsia."
        )
    else:
        system_prompt = (
            "You are a helpful assistant specialized in health information, with a focus on diabetes. "
            "Always respond in the same language as the user's question. Detect the user's language and reply in that language. "
            "Provide accurate, concise, and informative responses based on the given context. "
            "If the question is not related to health or diabetes, politely inform the user that you can only provide information on health and diabetes."
        )
    messages = [{"role": "system", "content": system_prompt}]
    for entry in conversation_history:
        if entry['query']:
            messages.append({"role": "user", "content": entry['query']})
        if entry['response']:
            messages.append({"role": "assistant", "content": entry['response']})
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=350,  # Increased for more complete answers
        temperature=0.7
    )
    answer = response.choices[0].message['content'].strip()
    import re
    # Convert markdown numbered lists to bullet points
    answer = re.sub(r'\n\d+\. ', '\n- ', answer)
    # Remove markdown bold (**) for plain chat
    answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
    # Add double newline before each bullet for clear separation
    answer = re.sub(r'\s*- ', '\n\n- ', answer)
    # Add a newline after periods followed by a capital letter (for paragraph breaks)
    answer = re.sub(r'(\.[ ]+)([A-Z])', r'\1\n\2', answer)
    # Remove extra newlines at the start
    answer = answer.lstrip('\n')
    # Ensure the answer ends with a complete sentence
    if not answer.endswith(('.', '!', '?')):
        answer += '.'
    return answer


# Load both datasets and embeddings with correct paths
data_diabetes = load_cleaned_data('data_diabetes/cleaned_diabetes.txt')
embeddings_diabetes = load_embeddings('data_diabetes/embeddings_diabetes.npy')

# Preeclampsia data/embeddings
try:
    data_preeclampsia = load_cleaned_data('data_preelampsia/cleaned_preecampsia.txt')
    embeddings_preeclampsia = load_embeddings('data_preelampsia/embeddings_preeclampsia.npy')
except Exception:
    data_preeclampsia = []
    embeddings_preeclampsia = None


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/ask', methods=['POST'])
def ask():
    import os
    from flask import url_for
    start_time = time.time()
    user_query = request.form.get('query', '')
    model = request.form.get('model', 'diabetes')
    audio_file = request.files.get('audio')
    image_file = request.files.get('image')

    # Handle audio: transcribe to text if present
    if audio_file:
        # Save audio temporarily
        audio_path = os.path.join('static', 'temp_audio.webm')
        audio_file.save(audio_path)
        # Transcribe using OpenAI Whisper
        with open(audio_path, 'rb') as af:
            transcript = openai.Audio.transcribe('whisper-1', af)
        user_query = transcript['text']

    # Namespace conversation history per model
    if 'conversation_history' not in session or not isinstance(session['conversation_history'], dict):
        session['conversation_history'] = {}
    conversation_history_dict = session['conversation_history']
    if model not in conversation_history_dict:
        conversation_history_dict[model] = []
    session['conversation_history'] = conversation_history_dict  # Re-assign to ensure Flask notices the change

    # Restrict cross-specialty questions
    preeclampsia_keywords = ['preeclampsia', 'pre-eclampsia', 'pre eclampsia', 'high blood pressure in pregnancy', 'maternal hypertension']
    diabetes_keywords = ['diabetes', 'gestational diabetes', 'blood sugar', 'glucose', 'insulin']

    def translate_if_needed(message, user_query):
        # Use langdetect to detect language
        from langdetect import detect
        import openai
        import re
        try:
            lang = detect(user_query)
        except Exception:
            lang = 'en'
        print(f"[DEBUG] Detected language: {lang}")
        # Always attempt translation, even if lang == 'en', to let OpenAI decide
        prompt = (
            f"Translate the following message to the language of the user question. "
            f"If the user question is in Luganda, translate to Luganda. "
            f"If the user question is in English, return the message as is.\n"
            f"User question: {user_query}\nMessage: {message}"
        )
        try:
            result = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are a translation assistant."}, {"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            translated = result.choices[0].message['content'].strip()
            # Remove any leading 'Translation:' or similar
            translated = re.sub(r'^Translation: *', '', translated, flags=re.I)
            return translated
        except Exception as e:
            print(f"[DEBUG] Translation error: {e}")
            return message

    if model == 'diabetes':
        if any(word in user_query.lower() for word in preeclampsia_keywords):
            response = (
                "You asked about preeclampsia.\n"
                "- Preeclampsia is a pregnancy complication involving high blood pressure.\n"
                "- For full details and support, please switch to the Pre-eclampsia model above."
            )
            response = translate_if_needed(response, user_query)
            conversation_history = conversation_history_dict[model]
            conversation_history.append({'query': user_query, 'response': response})
            conversation_history_dict[model] = conversation_history
            session['conversation_history'] = conversation_history_dict
            return jsonify({'response': response, 'conversation_history': conversation_history_dict[model]})
    if model == 'preeclampsia':
        if any(word in user_query.lower() for word in diabetes_keywords):
            response = (
                "You asked about diabetes.\n"
                "- Diabetes affects blood sugar levels and can occur during pregnancy (gestational diabetes).\n"
                "- For full details and support, please switch to the Gestational Diabetes model above."
            )
            response = translate_if_needed(response, user_query)
            conversation_history = conversation_history_dict[model]
            conversation_history.append({'query': user_query, 'response': response})
            conversation_history_dict[model] = conversation_history
            session['conversation_history'] = conversation_history_dict
            return jsonify({'response': response, 'conversation_history': conversation_history_dict[model]})

    # Select dataset and embeddings based on model, with error handling
    if model == 'preeclampsia':
        if data_preeclampsia and embeddings_preeclampsia is not None:
            context = retrieve_context(user_query, embeddings_preeclampsia, data_preeclampsia)
        else:
            return jsonify({'response': 'Preeclampsia model data is not available. Please select another model.', 'conversation_history': conversation_history_dict.get(model, [])})
    else:
        if data_diabetes and embeddings_diabetes is not None:
            context = retrieve_context(user_query, embeddings_diabetes, data_diabetes)
        else:
            return jsonify({'response': 'Diabetes model data is not available. Please contact the administrator.', 'conversation_history': conversation_history_dict.get(model, [])})


    # Generate response based on the entire conversation history for the selected model
    conversation_history = conversation_history_dict[model]
    conversation_history.append({'query': user_query, 'response': ''})  # Add the new query to the history for context generation
    response = generate_response_with_openai(conversation_history, model)
    conversation_history[-1]['response'] = response  # Update the latest entry with the generated response

    # Synthesize response to audio if audio was sent
    audio_url = None
    if audio_file:
        # Use OpenAI TTS (text-to-speech)
        tts_response = openai.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=response
        )
        audio_out_path = os.path.join('static', 'response_audio.mp3')
        with open(audio_out_path, 'wb') as f:
            f.write(tts_response.content)
        audio_url = url_for('static', filename='response_audio.mp3')

    # Update conversation history in the session
    conversation_history_dict[model] = conversation_history
    session['conversation_history'] = conversation_history_dict
    response_time = time.time()

    print(f"Time to get response from OpenAI: {response_time - start_time} seconds")
    print(f"Conversation History for {model}: {conversation_history_dict[model]}")  # Debug statement

    return jsonify({'response': response, 'conversation_history': conversation_history_dict[model], 'audio_url': audio_url})

if __name__ == '__main__':
    # # Load precomputed data and embeddings
    # data = load_cleaned_data('cleaned_data.txt')
    # embeddings = load_embeddings('embeddings.npy')
    
    # Use environment variable for port (for deployment platforms like Render)
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
