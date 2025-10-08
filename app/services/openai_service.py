"""
OpenAI Service for HealthMama AI
Handles communication with OpenAI API
"""
import openai
import logging
from typing import List, Dict, Optional
from config.settings import Config


class ConversationManager:
    """Manages conversation history for different models"""
    
    def __init__(self):
        self.conversations = {
            'diabetes': [],
            'preeclampsia': []
        }
        self.logger = logging.getLogger(__name__)
    
    def add_message(self, model: str, role: str, content: str):
        """Add a message to conversation history"""
        if model not in self.conversations:
            self.conversations[model] = []
        
        self.conversations[model].append({
            "role": role,
            "content": content
        })
        
        # Keep conversation history manageable
        if len(self.conversations[model]) > Config.MAX_CONVERSATION_HISTORY:
            self.conversations[model] = self.conversations[model][-Config.CONVERSATION_TRIM_SIZE:]
    
    def get_conversation(self, model: str) -> List[Dict[str, str]]:
        """Get conversation history for a model"""
        return self.conversations.get(model, [])
    
    def clear_conversation(self, model: str):
        """Clear conversation history for a model"""
        if model in self.conversations:
            self.conversations[model] = []


class OpenAIService:
    """Service for OpenAI API interactions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversation_manager = ConversationManager()
        
        # Set OpenAI API key
        openai.api_key = Config.OPENAI_API_KEY
    
    def generate_response(self, user_message: str, model: str = 'diabetes', context: Optional[List[str]] = None) -> str:
        """Generate response using OpenAI API"""
        try:
            # Add user message to conversation
            self.conversation_manager.add_message(model, "user", user_message)
            
            # Add context if provided
            if context:
                context_text = "\n".join(context[:3])
                self.conversation_manager.add_message(model, "system", f"Relevant context: {context_text}")
            
            # Get conversation history
            conversation_history = self.conversation_manager.get_conversation(model)
            
            # Detect language and add appropriate instructions
            conversation_history = self._detect_luganda_and_add_instruction(
                user_message, conversation_history, model
            )
            
            # Prepare messages for OpenAI
            messages = [{"role": "system", "content": self._get_system_prompt(model)}] + conversation_history
            
            # Make API call
            response = openai.ChatCompletion.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                max_tokens=Config.OPENAI_MAX_TOKENS,
                temperature=Config.OPENAI_TEMPERATURE
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Add AI response to conversation
            self.conversation_manager.add_message(model, "assistant", ai_response)
            
            return ai_response
            
        except openai.error.AuthenticationError:
            return "Error: Invalid OpenAI API key. Please check your configuration."
        except openai.error.RateLimitError:
            return "Error: OpenAI API rate limit exceeded. Please try again later."
        except openai.error.APIError as e:
            return f"Error: OpenAI API error - {str(e)}"
        except Exception as e:
            self.logger.error(f"OpenAI error: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _get_system_prompt(self, model: str) -> str:
        """Get system prompt for the model"""
        if model == 'preeclampsia':
            return (
                "You are a helpful assistant specialized in health information, with a focus on preeclampsia and maternal health. "
                "Always respond in the same language as the user's question. Detect the user's language and reply in that language. "
                "If the user asks in Luganda, provide detailed, culturally appropriate responses using simple, clear Luganda. "
                "Include practical advice that is relevant to Ugandan healthcare context. "
                "Use common Luganda health terms and explain medical concepts in ways that are easily understood. "
                "Provide accurate, concise, and informative responses based on the given context. "
                "If the question is not related to health or preeclampsia, politely inform the user that you can only provide information on health and preeclampsia."
            )
        else:  # diabetes
            return (
                "You are a helpful assistant specialized in health information, with a focus on diabetes and blood sugar management. "
                "Always respond in the same language as the user's question. Detect the user's language and reply in that language. "
                "If the user asks in Luganda, provide detailed, culturally appropriate responses using simple, clear Luganda. "
                "Include practical advice about diet, exercise, and diabetes management that is relevant to Ugandan lifestyle and available foods. "
                "Use common Luganda health terms and explain medical concepts in ways that are easily understood. "
                "Mention local foods like matooke, posho, beans, and their effects on blood sugar when relevant. "
                "Provide accurate, concise, and informative responses based on the given context. "
                "If the question is not related to health or diabetes, politely inform the user that you can only provide information on health and diabetes."
            )
    
    def _detect_luganda_and_add_instruction(self, user_message: str, conversation_history: List[Dict], model: str) -> List[Dict]:
        """Detect Luganda language and add specific instructions for better responses"""
        luganda_indicators = [
            'nga', 'bwe', 'ku', 'mu', 'nti', 'kiki', 'ani', 'wa', 'gw', 'ly', 'gy', 
            'ssukali', 'omusujja', 'obulwadde', 'emmere', 'amazzi'
        ]
        
        if any(indicator in user_message.lower() for indicator in luganda_indicators):
            conversation_history.append({
                "role": "system", 
                "content": (
                    "The user is asking in Luganda. Please respond in clear, detailed Luganda using simple terms. "
                    "Include practical advice relevant to Ugandan context. Use common Luganda health vocabulary "
                    "and explain medical terms clearly. Give specific examples with local foods like matooke, "
                    "posho, beans when discussing diet."
                )
            })
        
        return conversation_history
    
    def get_conversation_history(self, model: str) -> List[Dict[str, str]]:
        """Get conversation history for a model"""
        return self.conversation_manager.get_conversation(model)
    
    def clear_conversation_history(self, model: str):
        """Clear conversation history for a model"""
        self.conversation_manager.clear_conversation(model)