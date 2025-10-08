"""
Health check and monitoring routes
"""
from flask import Blueprint, jsonify, current_app

health_bp = Blueprint('health', __name__)


@health_bp.route('/health')
def health_check():
    """Health check endpoint for deployment platforms"""
    try:
        data_stats = current_app.data_service.get_data_stats()
        
        return jsonify({
            'status': 'healthy',
            'message': 'HealthMama AI is running',
            'version': '2.0',
            'data_loaded': data_stats
        }), 200
    except Exception as e:
        current_app.logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'message': f'Health check failed: {str(e)}'
        }), 500


@health_bp.route('/status')
def status():
    """Detailed status endpoint"""
    try:
        data_stats = current_app.data_service.get_data_stats()
        
        return jsonify({
            'status': 'running',
            'application': 'HealthMama AI',
            'version': '2.0',
            'environment': current_app.config.get('ENV', 'unknown'),
            'debug_mode': current_app.debug,
            'data_status': {
                'diabetes_loaded': data_stats['diabetes_entries'] > 0,
                'preeclampsia_loaded': data_stats['preeclampsia_entries'] > 0,
                'total_entries': data_stats['total_entries']
            },
            'configuration': {
                'openai_configured': bool(current_app.config.get('OPENAI_API_KEY')),
                'max_conversation_history': current_app.config.get('MAX_CONVERSATION_HISTORY', 10)
            }
        }), 200
    except Exception as e:
        current_app.logger.error(f"Status check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Status check failed: {str(e)}'
        }), 500


@health_bp.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """Test endpoint for debugging"""
    try:
        api_key = current_app.config.get('OPENAI_API_KEY')
        data_stats = current_app.data_service.get_data_stats()
        
        return jsonify({
            'method': 'GET/POST',
            'endpoint': 'test',
            'status': 'working',
            'openai_key_configured': bool(api_key),
            'api_key_length': len(api_key) if api_key else 0,
            'api_key_preview': f"{api_key[:10]}..." if api_key else None,
            'data_status': {
                'diabetes_loaded': data_stats['diabetes_entries'] > 0,
                'preeclampsia_loaded': data_stats['preeclampsia_entries'] > 0
            }
        }), 200
    except Exception as e:
        current_app.logger.error(f"Test endpoint failed: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Test failed: {str(e)}'
        }), 500