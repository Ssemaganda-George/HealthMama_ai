"""
API routes for HealthMama AI
Chat and messaging endpoints
"""
from flask import Blueprint, request, jsonify, current_app
from app.services.openai_service import OpenAIService
import time

api_bp = Blueprint('api', __name__)

# Initialize OpenAI service
openai_service = OpenAIService()


@api_bp.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        start_time = time.time()
        
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'response': 'Please send JSON data.'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data received',
                'response': 'Please provide valid JSON.'
            }), 400
        
        user_message = data.get('message', '').strip()
        model = data.get('model', 'diabetes')
        language = data.get('language', 'en')  # Get user's language
        
        if not user_message:
            return jsonify({
                'response': 'Please provide a message.',
                'conversation_history': openai_service.get_conversation_history(model),
                'audio_url': None
            })
        
        # Get relevant context from data service
        context = current_app.data_service.search_data(user_message, model, limit=5)
        
        # Generate response with language preference
        response = openai_service.generate_response(user_message, model, context, language)
        
        response_time = time.time() - start_time
        current_app.logger.info(f"Chat response generated in {response_time:.2f} seconds")
        
        return jsonify({
            'response': response,
            'conversation_history': openai_service.get_conversation_history(model),
            'audio_url': None,
            'response_time': round(response_time, 2)
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in chat endpoint: {str(e)}")
        model = request.get_json().get('model', 'diabetes') if request.is_json else 'diabetes'
        return jsonify({
            'error': f'Chat error: {str(e)}',
            'response': 'Sorry, I encountered an error processing your request.',
            'conversation_history': openai_service.get_conversation_history(model),
            'audio_url': None
        }), 500


@api_bp.route('/ask', methods=['POST'])
def ask():
    """Legacy endpoint for backward compatibility with frontend FormData"""
    try:
        # Handle both FormData and JSON
        if request.form:
            user_message = request.form.get('query', '').strip()
            model = request.form.get('model', 'diabetes')
            language = request.form.get('language', 'en')
        elif request.is_json:
            data = request.get_json()
            user_message = data.get('message', '').strip()
            model = data.get('model', 'diabetes')
            language = data.get('language', 'en')
        else:
            return jsonify({
                'error': 'No data received',
                'response': 'Please provide valid data.'
            }), 400
        
        if not user_message:
            return jsonify({
                'response': 'Please provide a message.',
                'conversation_history': openai_service.get_conversation_history(model),
                'audio_url': None
            })
        
        # Get relevant context from data service
        context = current_app.data_service.search_data(user_message, model, limit=5)
        
        # Generate response with language preference
        response = openai_service.generate_response(user_message, model, context, language)
        
        return jsonify({
            'response': response,
            'conversation_history': openai_service.get_conversation_history(model),
            'audio_url': None
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in ask endpoint: {str(e)}")
        model = 'diabetes'
        if request.form:
            model = request.form.get('model', 'diabetes')
        elif request.is_json:
            model = request.get_json().get('model', 'diabetes')
        
        return jsonify({
            'error': f'Ask error: {str(e)}',
            'response': 'Sorry, I encountered an error processing your request.',
            'conversation_history': openai_service.get_conversation_history(model),
            'audio_url': None
        }), 500


@api_bp.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history for a model"""
    try:
        data = request.get_json() if request.is_json else {}
        model = data.get('model', 'diabetes')
        
        openai_service.clear_conversation_history(model)
        
        return jsonify({
            'message': f'Conversation history cleared for {model}',
            'model': model
        })
        
    except Exception as e:
        current_app.logger.error(f"Error clearing history: {str(e)}")
        return jsonify({
            'error': f'Error clearing history: {str(e)}'
        }), 500


@api_bp.route('/models', methods=['GET'])
def get_models():
    """Get available models"""
    return jsonify({
        'models': [
            {
                'id': 'diabetes',
                'name': 'Gestational Diabetes',
                'description': 'Specialized in diabetes and blood sugar management'
            },
            {
                'id': 'preeclampsia',
                'name': 'Pre-eclampsia',
                'description': 'Specialized in preeclampsia and maternal health'
            }
        ]
    })