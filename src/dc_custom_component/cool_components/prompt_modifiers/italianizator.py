from typing import Dict, List
from haystack import component
from haystack.dataclasses import ChatMessage


@component
class Italianizator:
    """A component that italianize 🤌 a given prompt"""

    system_prompt = (
        "You are an helpful assistant.\n"
        "Your goal is to answer to a user question.\n"
        "Instructions:\n"
        "* Start the conversation with 'Ciao!'\n"
        "* Always respond with the same language used in the question.\n"
        "* Always conclude the answer with an Italian quote about food, love or Italy itself.\n"
        "* Use emoji if needed!\n\n"
        "Example:\n"
        "Question: what is the capital of Brazil?\n"
        "Answer: Ciao! The capital of Brazil is Brasilia! 🇧🇷 And as we say in Italy: *Il buon giorno si vede dal mattino* 😊\n\n"
    )

    @component.output_types(italianized_chat_prompt=List[ChatMessage])
    def run(
        self, chat_prompt_template: List[ChatMessage]
    ) -> Dict[str, List[ChatMessage]]:
        """
        Modify the original chat system prompt template by adding custom instructions
        """

        user_message = [x for x in chat_prompt_template if x.role == "user"][0]

        italianized_chat_prompt = [
            ChatMessage.from_system(self.system_prompt),
            user_message,
        ]

        return {"italianized_chat_prompt": italianized_chat_prompt}
