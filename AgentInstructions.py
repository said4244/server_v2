from livekit.agents import Agent
import logging
import os
logger = logging.getLogger(__name__)
class DebugAvatarAgent(Agent):
    """Agent with debug logging and system identification"""
    
    def __init__(self, system_type="unknown") -> None:
        avatar_language = os.getenv("AVATAR_LANGUAGE")
        avatar_language_stt = os.getenv("AVATAR_LANGUAGE_STT")
        if not avatar_language:
            logger.warning("AVATAR_LANGUAGE environment variable is not set. Defaulting to 'ar'.")
            avatar_language = "ar"
            language = "arabic-Syrian"
            
        elif avatar_language == "ar":
            language = "arabic-Syrian"
        elif avatar_language == "fr":
            language = "french" 

        elif avatar_language == "du":
            language = "German"
        else:
            language = "english"
        if system_type == "elevenlabs":
            instructions = f"You're the LLM between the STT and TTS systems, and your output will be sent to ElevenLabs v2 multilingual TTS, make full use of its capabilities to make the speech as realistic and natural as possible. Always respond in {language} language. Be friendly and conversational. If I am asking you to respond in a particular dialect, I want you to enforce this really hard, to make sure the phonetics and pronuncation are accurate to that dialect."
        elif system_type == "openai_realtime":
            instructions = f"Instruction for the LLM in the openai realtime API, and your output will be sent to the TTS, make full use of its capabilities to make the speech as realistic and natural as possible. Always respond in {language} language. Be friendly and conversational. If I am asking you to respond in a particular dialect, I want you to enforce this really hard, to make sure the phonetics and pronuncation are accurate to that dialect."
        else:
            instructions = f"You're the LLM between the STT and TTS systems, and your output will be sent to the TTS, make full use of its capabilities to make the speech as realistic and natural as possible. Always respond in {language} language. Be friendly and conversational. If I am asking you to respond in a particular dialect, I want you to enforce this really hard, to make sure the phonetics and pronuncation are accurate to that dialect."

        super().__init__(instructions=instructions)
        self.system_type = system_type
        self.avatar_language = avatar_language
        logger.info(f"DebugAvatarAgent initialized with system type: {system_type}, language: {avatar_language}")
    
    def get_greeting_message(self) -> str:
        """Return appropriate greeting message based on system type and language"""
        if self.avatar_language == "ar":
            if self.system_type == "elevenlabs":
                return "مرحبا! أنا مساعد ذكي أتحدث العربية باستخدام تقنية إيليفن لابز . كيف يمكنني مساعدتك اليوم؟"
            elif self.system_type == "openai_realtime":
                return "مرحبا! أنا مساعد ذكي أتحدث العربية باستخدام تقنية أوبن آي . كيف يمكنني مساعدتك اليوم؟"
            else:
                return "مرحبا! أنا مساعد ذكي أتحدث العربية. كيف يمكنني مساعدتك اليوم؟"
        elif self.avatar_language == "fr":
            if self.system_type == "elevenlabs":
                return "Salut! Je suis un assistant IA parlant français avec la voix ElevenLabs. Comment puis-je vous aider aujourd'hui?"
            elif self.system_type == "openai_realtime":
                return "Salut! Je suis un assistant IA propulsé par l'API OpenAI Realtime. Comment puis-je vous aider aujourd'hui?"
            else:
                return "Salut! Je suis un assistant IA avec un avatar visuel. Comment puis-je vous aider aujourd'hui?"
        elif self.avatar_language == "du":
            if self.system_type == "elevenlabs":
                return "Hallo! Ich bin ein KI-Assistent, der Deutsch mit der ElevenLabs-Stimme spricht. Wie kann ich Ihnen heute helfen?"
            elif self.system_type == "openai_realtime":
                return "Hallo! Ich bin ein KI-Assistent, der von der OpenAI Realtime API unterstützt wird. Wie kann ich Ihnen heute helfen?"
            else:
                return "Hallo! Ich bin ein KI-Assistent mit einem visuellen Avatar. Wie kann ich Ihnen heute helfen?"
        else:  # Default to English
            if self.system_type == "elevenlabs":
                return "Hey! I'm an AI assistant speaking through your custom STS stack with ElevenLabs voice. How can I help you today?"
            elif self.system_type == "openai_realtime":
                return "Hey! I'm an AI assistant powered by OpenAI Realtime API. How can I help you today?"
            else:
                return "Hey! I'm an AI assistant with a visual avatar. How can I help you today?"
        self.system_type = system_type
        logger.info(f"DebugAvatarAgent initialized with system type: {system_type}")
