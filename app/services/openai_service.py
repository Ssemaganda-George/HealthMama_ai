"""
OpenAI Service for HealthMama AI
Handles communication with OpenAI API with enhanced multi-language support
"""
import openai
import logging
from typing import Dict, Any, Optional, List
from config.settings import Config


class LugandaMedicalTranslator:
    """Comprehensive Luganda medical terminology and response system"""
    
    @staticmethod
    def get_luganda_medical_terms():
        """Simple, everyday Luganda medical terms people actually use"""
        return {
            # Basic medical terms - simple and commonly used
            'diabetes': 'ssukali',
            'blood_sugar': 'ssukali mu musayi', 
            'high_blood_sugar': 'ssukali gweyongera',
            'medication': 'ddagala',
            'medicine': 'ddagala',
            
            # Simple pregnancy terms
            'pregnancy': 'lubuto',
            'pregnant': 'mu lubuto',
            'baby': 'mwana',
            'birth': 'kuzaala',
            'mother': 'maama',
            
            # Basic body parts
            'blood': 'musayi',
            'head': 'mutwe',
            'eyes': 'maaso',
            'stomach': 'lubuto', # also used for belly/stomach
            'body': 'mubiri',
            
            # Simple symptoms  
            'headache': 'mutwe gunuma',
            'pain': 'bulumi',
            'tired': 'koowa',
            'sick': 'mulwadde',
            
            # Basic food
            'food': 'mmere',
            'water': 'mazzi',
            'eat': 'lya',
            
            # Medical care
            'doctor': 'musawo',
            'hospital': 'ddwaliro'
        }
    
    @staticmethod
    def get_luganda_response_templates():
        """Simple, natural Luganda response patterns like real conversations"""
        return {
            'greeting_response': [
                "Webale. Oli otya?",
                "Nkusanyuse. Ndi wano okukuyamba.",
                "Webale okubuuza."
            ],
            
            'diabetes_simple': [
                "Ssukali gweyongera guyinza okukuleetera ebizibu.",
                "Ssukali mu musayi bweyongera si kirungi.",
                "Kyetaaga okugenda eri omusawo."
            ],
            
            'pregnancy_simple': [
                "Mu lubuto ssukali guyinza okweyongera.",
                "Omukazi mu lubuto yeetaaga okulabirirwa bulungi.",
                "Kyetaaga okugenda eri omusawo mangu."
            ],
            
            'advice_simple': [
                "Kino kye nkugamba:",
                "Kiki kyoyagala okukola:",
                "Obuyambi bwange:"
            ],
            
            'doctor_urgent': [
                "Genda eri omusawo mangu.",
                "Kyetaaga okugenda ku ddwaliro.",
                "Laga omusawo amangu ddala."
            ],
            
            'diet_simple': [
                "Ku mmere:",
                "Ebyo bye tolya:",
                "Ku kulya:"
            ],
            
            'common_closings': [
                "Kwatira obulungi.",
                "Labirira omubiri gwo.",
                "Webale."
            ]
        }
    
    @staticmethod
    def format_luganda_response(content_type: str, medical_advice: str, urgency: str = 'normal') -> str:
        """Format a medical response in proper Luganda with cultural context"""
        templates = LugandaMedicalTranslator.get_luganda_response_templates()
        
        # Select appropriate greeting based on content type
        if content_type == 'diabetes':
            intro = templates['diabetes_intro'][0]
        elif content_type == 'pregnancy':
            intro = templates['pregnancy_intro'][0]
        else:
            intro = templates['greeting_response'][0]
        
        # Format the main advice
        advice_prefix = templates['advice_prefix'][0]
        
        # Add urgency if needed
        urgency_text = ""
        if urgency == 'high':
            urgency_text = templates['doctor_recommendation'][0] + "\n\n"
        
        # Construct response with proper Luganda structure
        response = f"{intro}\n\n{advice_prefix}\n{medical_advice}\n\n{urgency_text}"
        
        # Add cultural closing
        response += "Obulamu bwo bukulu nnyo. Kuuma omubiri gwo obulungi."
        
        return response


class OpenAIService:
    """Enhanced OpenAI service with comprehensive Luganda support"""
    
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY
        self.logger = logging.getLogger(__name__)
        self.luganda_translator = LugandaMedicalTranslator()
        
        # Conversation history for each model
        self.diabetes_conversation = []
        self.preeclampsia_conversation = []
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
    
    def generate_response(self, user_message: str, model: str = 'diabetes', context: Optional[List[str]] = None, language: str = 'en') -> str:
        """Generate response using OpenAI API with enhanced language support"""
        try:
            # Check for cross-model queries first
            cross_model_response = self._check_cross_model_query(user_message, model, language)
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
            
            # Enhanced Luganda processing
            if language == 'lg':
                response = self._generate_enhanced_luganda_response(user_message, model, conversation_history)
            else:
                # Add language-specific instructions for other languages
                conversation_history = self._add_language_instruction(language, conversation_history, model)
                response = self._generate_standard_response(conversation_history)
            
            # Add assistant response to conversation
            self.conversation_manager.add_message(model, "assistant", response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(language, model)
    
    def _generate_enhanced_luganda_response(self, user_message: str, model: str, conversation_history: List[Dict]) -> str:
        """Generate enhanced Luganda response with proper medical terminology and cultural context"""
        try:
            # Prepare enhanced Luganda system prompt
            luganda_system_prompt = self._create_luganda_system_prompt(model)
            
            # Prepare conversation with enhanced Luganda instructions
            messages = [
                {"role": "system", "content": luganda_system_prompt}
            ]
            
            # Add conversation history with Luganda context
            for message in conversation_history[-6:]:  # Keep last 6 messages for context
                messages.append(message)
            
            # Generate response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=400,
                temperature=0.7
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Post-process the response for better Luganda
            enhanced_response = self._enhance_luganda_response(raw_response, model, user_message)
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Error generating Luganda response: {e}")
            return self._get_luganda_fallback(model, user_message)
    
    def _create_luganda_system_prompt(self, model: str) -> str:
        """Create simple, natural Luganda system prompt"""
        base_prompt = f"""You are a health helper who speaks Luganda naturally. Help people with {model} questions.

LUGANDA SPEAKING RULES:
- Use simple, everyday Luganda words
- Talk like a normal Luganda speaker would talk
- Don't use too many big words
- Be friendly and helpful
- Keep answers short and clear

SIMPLE WORDS TO USE:
- ssukali (sugar/diabetes)
- musayi (blood)
- mutwe (head) 
- mubiri (body)
- ddagala (medicine)
- musawo (doctor)
- ddwaliro (hospital)
- mmere (food)
- mazzi (water)
- lubuto (pregnancy)
- mwana (baby)

HOW TO TALK:
1. Say "Webale" or "Oli otya?"
2. Answer the question simply
3. Give helpful advice
4. Say to see a doctor if serious
5. End nicely like "Kwatira bulungi"

IMPORTANT: 
- Use normal Luganda, not fancy language
- Give short, helpful answers
- Don't make it complicated
- Be like talking to a friend"""

        if model == 'diabetes':
            base_prompt += "\n\nYou help with ssukali (diabetes) problems and pregnancy health."
        else:
            base_prompt += "\n\nYou help with high blood pressure in pregnancy and mother health."
            
        return base_prompt
    
    def _enhance_luganda_response(self, raw_response: str, model: str, user_message: str) -> str:
        """Keep Luganda response simple and natural"""
        # Only replace the most obvious English words that might slip through
        simple_replacements = {
            'diabetes': 'ssukali',
            'doctor': 'musawo', 
            'hospital': 'ddwaliro',
            'medicine': 'ddagala',
            'pregnancy': 'lubuto',
            'baby': 'mwana',
            'food': 'mmere',
            'water': 'mazzi'
        }
        
        enhanced_response = raw_response
        for english, luganda in simple_replacements.items():
            enhanced_response = enhanced_response.replace(english, luganda)
        
        # Only add greeting if completely missing
        if not any(greeting in enhanced_response.lower() for greeting in ['webale', 'oli']):
            enhanced_response = "Webale. " + enhanced_response
        
        # Simple, natural closing
        if not enhanced_response.endswith(('.', '!', '?')):
            enhanced_response += "."
            
        # Don't add complicated closings - keep it simple
        return enhanced_response
    
    def _get_luganda_fallback(self, model: str, user_message: str) -> str:
        """Simple fallback responses in natural Luganda"""
        if 'bulunji' in user_message.lower() or 'bulungi' in user_message.lower():
            return "Webale. Ndi bulungi. Gwe oli otya? Njagala okukuyamba ku ssukali. Kiki kyoyagala okumanya?"
        
        if model == 'diabetes':
            return "Nkusonyiwa. Ku ssukali, genda eri musawo amangu. Webale."
        else:
            return "Nkusonyiwa. Ku nsonga z'omukazi mu lubuto, genda eri musawo. Webale."
    
    def _generate_standard_response(self, conversation_history: List[Dict]) -> str:
        """Generate standard response for non-Luganda languages"""
        try:
            # Prepare messages for OpenAI
            messages = [{"role": "system", "content": self._get_system_prompt("diabetes")}] + conversation_history
            
            # Generate response using OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            return ai_response
            
        except Exception as e:
            self.logger.error(f"Error in standard response generation: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again or consult with a healthcare provider."
            
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
    
    def _check_cross_model_query(self, user_message: str, current_model: str, language: str = 'en') -> Optional[str]:
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
                    'diabetes', 'preeclampsia', user_message, message_lower, language
                )
        
        # Check if preeclampsia model is being asked about diabetes
        elif current_model == 'preeclampsia':
            if any(keyword in message_lower for keyword in diabetes_keywords):
                return self._generate_cross_model_response(
                    'preeclampsia', 'diabetes', user_message, message_lower, language
                )
        
        return None
    
    def _generate_cross_model_response(self, current_model: str, suggested_model: str, 
                                     user_message: str, message_lower: str, language: str = 'en') -> str:
        """Generate response when user asks about wrong topic"""
        
        if current_model == 'diabetes' and suggested_model == 'preeclampsia':
            if language == 'lg':  # Luganda
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
            elif language == 'sw':  # Swahili
                return (
                    "Naelewa kuwa unajibu kuhusu preeclampsia. Hii ni hali ya hatari kwa wanawake wajawazito. "
                    "Preeclampsia inamaanisha shinikizo la damu la juu na protini katika mkojo.\n\n"
                    "<strong>Dalili za Preeclampsia:</strong>\n"
                    "• Shinikizo la damu la juu (zaidi ya 140/90)\n"
                    "• Protini katika mkojo\n"
                    "• Uvimbe wa mikono, miguu, na uso\n"
                    "• Maumivu makali ya kichwa\n"
                    "• Mabadiliko ya macho\n\n"
                    "<strong>Hata hivyo,</strong> mimi ni mtaalamu wa kisukari (diabetes). "
                    "Kwa msaada kamili kuhusu preeclampsia, tafadhali <strong>badilisha kwa Pre-eclampsia Model</strong> "
                    "katika menyu hapo juu. Hiyo ni iliyoundwa maalum kutoa ushauri wa kina kuhusu preeclampsia."
                )
            elif language == 'fr':  # French
                return (
                    "Je comprends que vous posez des questions sur la prééclampsie. C'est une condition grave chez les femmes enceintes. "
                    "La prééclampsie signifie une tension artérielle élevée et des protéines dans l'urine.\n\n"
                    "<strong>Signes de Prééclampsie:</strong>\n"
                    "• Tension artérielle élevée (au-dessus de 140/90)\n"
                    "• Protéines dans l'urine\n"
                    "• Gonflement des mains, pieds et visage\n"
                    "• Maux de tête sévères\n"
                    "• Changements de vision\n\n"
                    "<strong>Cependant,</strong> je suis spécialisé dans le diabète. "
                    "Pour une aide complète sur la prééclampsie, veuillez <strong>passer au Modèle Prééclampsie</strong> "
                    "en utilisant le menu déroulant ci-dessus. Ce modèle est spécialement conçu pour fournir des conseils détaillés sur la prééclampsie."
                )
            else:  # English default
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
            if language == 'lg':  # Luganda
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
            elif language == 'sw':  # Swahili
                return (
                    "Naelewa kuwa unajibu kuhusu kisukari. Hii ni hali ambapo kiwango cha sukari damu "
                    "kinakuwa juu sana, na inaweza kuwa hatari wakati wa uja uzito.\n\n"
                    "<strong>Dalili za Kisukari wakati wa Uja uzito:</strong>\n"
                    "• Kiwango cha sukari damu cha juu\n"
                    "• Kunywa maji mengi\n"
                    "• Kukojoa mara nyingi\n"
                    "• Kupoteza uzito bila sababu\n"
                    "• Uchovu mkubwa\n\n"
                    "<strong>Hata hivyo,</strong> mimi ni mtaalamu wa preeclampsia na afya ya mama mjamzito. "
                    "Kwa msaada kamili kuhusu kisukari, tafadhali <strong>badilisha kwa Gestational Diabetes Model</strong> "
                    "katika menyu hapo juu. Hiyo ni iliyoundwa maalum kutoa ushauri wa kina kuhusu kisukari na chakula."
                )
            elif language == 'fr':  # French
                return (
                    "Je comprends que vous posez des questions sur le diabète. C'est une condition où le taux de sucre dans le sang "
                    "devient trop élevé, et cela peut être particulièrement important pendant la grossesse.\n\n"
                    "<strong>Signes clés du Diabète:</strong>\n"
                    "• Taux de sucre dans le sang élevé\n"
                    "• Soif accrue\n"
                    "• Mictions fréquentes\n"
                    "• Perte de poids inexpliquée\n"
                    "• Fatigue accrue\n\n"
                    "<strong>Cependant,</strong> je suis spécialisé dans la prééclampsie et les complications de santé maternelle. "
                    "Pour une aide complète avec le diabète, veuillez <strong>passer au Modèle Diabète Gestationnel</strong> "
                    "en utilisant le menu déroulant ci-dessus. Ce modèle est spécialement conçu pour fournir "
                    "des conseils détaillés sur la gestion du sucre dans le sang, l'alimentation et les soins du diabète."
                )
            else:  # English default
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
                "LANGUAGE INSTRUCTIONS: Always respond in English by default, as this is your primary language. "
                "Only respond in another language (like Luganda) if the user explicitly asks their question in that language "
                "and it's clear they prefer that language. If a user asks in English, always respond in English. "
                "When responding in Luganda, provide detailed, culturally appropriate responses using simple, clear Luganda. "
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
                "LANGUAGE INSTRUCTIONS: Always respond in English by default, as this is your primary language. "
                "Only respond in another language (like Luganda) if the user explicitly asks their question in that language "
                "and it's clear they prefer that language. If a user asks in English, always respond in English. "
                "When responding in Luganda, provide detailed, culturally appropriate responses using simple, clear Luganda. "
                "Include practical advice about diet, exercise, and diabetes management that is relevant to Ugandan lifestyle and available foods. "
                "Use common Luganda health terms and explain medical concepts in ways that are easily understood. "
                "Mention local foods like matooke, posho, beans, and their effects on blood sugar when relevant. "
                "Provide accurate, concise, and informative responses based on your specialized knowledge of diabetes."
            )
    
    def _add_language_instruction(self, language: str, conversation_history: List[Dict], model: str) -> List[Dict]:
        """Add language-specific instructions for better responses"""
        
        language_instructions = {
            'lg': (  # Luganda
                "The user is asking in Luganda. Please respond in clear, detailed Luganda using simple terms. "
                "Include practical advice relevant to Ugandan context. Use common Luganda health vocabulary "
                "and explain medical terms clearly. Give specific examples with local foods like matooke, "
                "posho, beans when discussing diet. Be culturally sensitive and use respectful Luganda language."
            ),
            'sw': (  # Swahili
                "The user is asking in Swahili. Please respond in clear, detailed Swahili using simple terms. "
                "Include practical advice relevant to East African context. Use common Swahili health vocabulary "
                "and explain medical terms clearly. Give specific examples with local foods and cultural practices "
                "when relevant. Be culturally sensitive and use respectful Swahili language."
            ),
            'fr': (  # French
                "The user is asking in French. Please respond in clear, detailed French using simple terms. "
                "Include practical advice and explain medical terms clearly. Use proper French medical vocabulary "
                "while keeping the language accessible and easy to understand. Be professional and respectful."
            ),
            'en': (  # English (default)
                "The user is asking in English. Please respond in clear, professional English. "
                "Provide detailed, medically accurate information while keeping it accessible and easy to understand."
            )
        }
        
        # Use the appropriate instruction based on detected language
        instruction = language_instructions.get(language, language_instructions['en'])
        
        conversation_history.append({
            "role": "system", 
            "content": instruction
        })
        
        return conversation_history
    
    def get_conversation_history(self, model: str) -> List[Dict[str, str]]:
        """Get conversation history for a model"""
        return self.conversation_manager.get_conversation(model)
    
    def clear_conversation_history(self, model: str):
        """Clear conversation history for a model"""
        self.conversation_manager.clear_conversation(model)