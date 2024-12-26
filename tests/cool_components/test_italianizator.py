from dc_custom_component.cool_components.prompt_modifiers.italianizator import (
    Italianizator,
)
from haystack.dataclasses import ChatMessage
from typing import List, Dict


class TestItalianizator:
    def test_prompt_template_length(self) -> None:
        italianizator: Italianizator = Italianizator()

        template = [
            ChatMessage.from_system("generic system prompt"),
            ChatMessage.from_user(
                "Question: 'What is the capital of Italy?'\nAnswer: "
            ),
        ]

        result: Dict[str, List[ChatMessage]] = italianizator.run(
            chat_prompt_template=template
        )

        assert len(result["italianized_chat_prompt"]) == 2

    def test_system_prompt(self) -> None:
        italianizator: Italianizator = Italianizator()

        template = [
            ChatMessage.from_system("generic system prompt"),
            ChatMessage.from_user(
                "Question: 'What is the capital of Italy?'\nAnswer: "
            ),
        ]

        result: Dict[str, List[ChatMessage]] = italianizator.run(
            chat_prompt_template=template
        )

        system_message = [
            x for x in result["italianized_chat_prompt"] if x.role == "system"
        ][0]

        assert system_message.content == italianizator.system_prompt
