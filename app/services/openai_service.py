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
            # Check for cross-model queries first
            cross_model_response = self._check_cross_model_query(user_message, model)
            if cross_model_response:
                # Add both user message and assistant response to conversation
                self.conversation_manager.add_message(model, "user", user_message)
                self.conversation_manager.add_message(model, "assistant", cross_model_response)
                return cross_model_response
            
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
    
    def _check_cross_model_query(self, user_message: str, current_model: str) -> Optional[str]:
        """Check if user is asking about the wrong topic for current model"""
        message_lower = user_message.lower()
        
        # Define keywords for each condition
        diabetes_keywords = [
            'diabetes', 'diabetic', 'blood sugar', 'glucose', 'insulin', 'glycemic', 
            'hyperglycemia', 'hypoglycemia', 'gestational diabetes', 'blood glucose',
            'sugar level', 'a1c', 'hemoglobin a1c', 'metformin', 'ssukali', 'sukali'
        ]
        
        preeclampsia_keywords = [
            'preeclampsia', 'pre-eclampsia', 'eclampsia', 'high blood pressure', 
            'hypertension', 'protein in urine', 'proteinuria', 'swelling', 'edema',
            'seizures', 'headaches', 'vision changes', 'blood pressure', 'bp',
            'pregnancy complications', 'maternal health', 'pregnancy hypertension'
        ]
        
        # Check if diabetes model is being asked about preeclampsia
        if current_model == 'diabetes':
            if any(keyword in message_lower for keyword in preeclampsia_keywords):
                return self._generate_cross_model_response(
                    'diabetes', 'preeclampsia', user_message, message_lower
                )
        
        # Check if preeclampsia model is being asked about diabetes
        elif current_model == 'preeclampsia':
            if any(keyword in message_lower for keyword in diabetes_keywords):
                return self._generate_cross_model_response(
                    'preeclampsia', 'diabetes', user_message, message_lower
                )
        
        return None
    
    def _generate_cross_model_response(self, current_model: str, suggested_model: str, 
                                     user_message: str, message_lower: str) -> str:
        """Generate response when user asks about wrong topic"""
        
        # Detect if user is asking in Luganda
        luganda_indicators = ['nga', 'bwe', 'ku', 'mu', 'nti', 'kiki', 'ani', 'wa', 'gw', 'ly', 'gy']
        is_luganda = any(indicator in message_lower for indicator in luganda_indicators)
        
        if current_model == 'diabetes' and suggested_model == 'preeclampsia':
            if is_luganda:
                return (
                    "Nkutegeera nti obuuza ku preeclampsia. Kino kiragiro kya maanyi nnyo ku bakazi abali mu lubuto. "
                    "Preeclampsia kitegeeza omusujja gw'omusaayi ogweyongera n'amafuta mu nsigo.\n\n"
                    "<strong>Obubonero bwa Preeclampsia:</strong>\n"
                    "• Omusujja gw'omusaayi ogweyongera (okusukka 140/90)\n"
                    "• Amafuta mu nsigo\n"
                    "• Okuzimba mu mikono, amagulu, n'amaaso\n"
                    "• Omutwe okukuba ennyo\n"
                    "• Okulaba obutali bulungi\n\n"
                    "<strong>Kyokka,</strong> nze ndi mubunyangaanya bw'obuwuka bwa ssukali (diabetes). "
                    "Okufuna obuyambi obujjuvu ku preeclampsia, <strong>kyuusakyuuse ku Pre-eclampsia Model</strong> "
                    "mu menu waggulu. Eyo gye munaafunira obuyambi obwetongole ku preeclampsia n'obulamu bw'omukazi ali mu lubuto."
                )
            else:
                return (
                    "I understand you're asking about preeclampsia. This is a serious pregnancy condition "
                    "characterized by high blood pressure and protein in the urine.\n\n"
                    "<strong>Key signs of Preeclampsia:</strong>\n"
                    "• High blood pressure (above 140/90)\n"
                    "• Protein in urine\n"
                    "• Swelling in hands, feet, and face\n"
                    "• Severe headaches\n"
                    "• Vision changes\n\n"
                    "<strong>However,</strong> I'm specialized in diabetes and blood sugar management. "
                    "For comprehensive help with preeclampsia, please <strong>switch to the Pre-eclampsia Model</strong> "
                    "using the dropdown menu above. That model is specifically designed to provide "
                    "detailed guidance on preeclampsia and maternal health complications."
                )
        
        elif current_model == 'preeclampsia' and suggested_model == 'diabetes':
            if is_luganda:
                return (
                    "Nkutegeera nti obuuza ku diabetes/ssukali. Kino kiragiro kya ssukali mu musaayi "
                    "okweyongera ennyo, era kyandibadde kya maanyi nnyo mu bakazi abali mu lubuto.\n\n"
                    "<strong>Obubonero bwa Diabetes mu lubuto:</strong>\n"
                    "• Ssukali mu musaayi okweyongera\n"
                    "• Okunywa amazzi mangi\n"
                    "• Okukojjagana ennyo\n"
                    "• Okucwa obuzito mu bwangu\n"
                    "• Okweraliikirira mu bwangu\n\n"
                    "<strong>Kyokka,</strong> nze ndi mubunyangaanya bw'obuwuka bwa preeclampsia n'obulamu bw'omukazi. "
                    "Okufuna obuyambi obujjuvu ku diabetes, <strong>kyuusakyuuse ku Gestational Diabetes Model</strong> "
                    "mu menu waggulu. Eyo gye munaafunira obuyambi obwetongole ku ssukali n'emmere gy'olya."
                )
            else:
                return (
                    "I understand you're asking about diabetes. This is a condition where blood sugar levels "
                    "become too high, and it can be particularly important during pregnancy.\n\n"
                    "<strong>Key signs of Diabetes:</strong>\n"
                    "• High blood sugar levels\n"
                    "• Increased thirst\n"
                    "• Frequent urination\n"
                    "• Unexplained weight loss\n"
                    "• Increased fatigue\n\n"
                    "<strong>However,</strong> I'm specialized in preeclampsia and maternal health complications. "
                    "For comprehensive help with diabetes, please <strong>switch to the Gestational Diabetes Model</strong> "
                    "using the dropdown menu above. That model is specifically designed to provide "
                    "detailed guidance on blood sugar management, diet, and diabetes care."
                )
        
        return ""
    
    def _get_system_prompt(self, model: str) -> str:
        """Get system prompt for the model"""
        if model == 'preeclampsia':
            return (
                "You are a helpful assistant specialized in preeclampsia and maternal health complications. "
                "Your primary expertise is in preeclampsia, high blood pressure during pregnancy, maternal health, "
                "pregnancy complications, and related conditions.\n\n"
                "IMPORTANT: You are NOT specialized in diabetes. If users ask about diabetes, blood sugar, "
                "or insulin-related questions, provide only basic information and clearly direct them to "
                "switch to the Gestational Diabetes Model.\n\n"
                "Always respond in the same language as the user's question. Detect the user's language and reply in that language. "
                "If the user asks in Luganda, provide detailed, culturally appropriate responses using simple, clear Luganda. "
                "Include practical advice that is relevant to Ugandan healthcare context. "
                "Use common Luganda health terms and explain medical concepts in ways that are easily understood. "
                "Provide accurate, concise, and informative responses based on your specialized knowledge of preeclampsia."
            )
        else:  # diabetes
            return (
                "You are a helpful assistant specialized in diabetes, gestational diabetes, and blood sugar management. "
                "Your primary expertise is in diabetes care, blood glucose monitoring, diet management, "
                "insulin therapy, and diabetes-related complications.\n\n"
                "IMPORTANT: You are NOT specialized in preeclampsia. If users ask about preeclampsia, "
                "high blood pressure during pregnancy, or maternal complications unrelated to diabetes, "
                "provide only basic information and clearly direct them to switch to the Pre-eclampsia Model.\n\n"
                "Always respond in the same language as the user's question. Detect the user's language and reply in that language. "
                "If the user asks in Luganda, provide detailed, culturally appropriate responses using simple, clear Luganda. "
                "Include practical advice about diet, exercise, and diabetes management that is relevant to Ugandan lifestyle and available foods. "
                "Use common Luganda health terms and explain medical concepts in ways that are easily understood. "
                "Mention local foods like matooke, posho, beans, and their effects on blood sugar when relevant. "
                "Provide accurate, concise, and informative responses based on your specialized knowledge of diabetes."
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