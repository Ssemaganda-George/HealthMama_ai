"""
Main routes for HealthMama AI
"""
from flask import Blueprint, render_template, current_app

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Main application page"""
    try:
        current_app.logger.info("Serving main index page")
        return render_template('index.html')
    except Exception as e:
        current_app.logger.error(f"Template error: {e}")
        return f"HealthMama AI is running, but template error: {str(e)}", 200


@main_bp.route('/about')
def about():
    """About page"""
    return render_template('about.html') if current_app.config.get('SHOW_ABOUT') else "About page not configured", 404