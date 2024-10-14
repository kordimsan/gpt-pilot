from typing import Optional

from langchain.chat_models.gigachat import GigaChat
from langchain.schema import HumanMessage, SystemMessage

from core.config import LLMProvider
from core.llm.convo import Convo
from core.log import get_logger

from .base import BaseLLMClient

log = get_logger(__name__)

# Maximum number of tokens supported by GigaChat Claude 3
MAX_TOKENS = 4096
MAX_TOKENS_SONNET = 8192


class GigaChatClient(BaseLLMClient):
    provider = LLMProvider.GIGACHAT

    def _init_client(self):
        self.client = GigaChat(
            credentials=self.config.api_key,
            scope="GIGACHAT_API_PERS",
            model=self.config.model,
            verify_ssl_certs=False,
            streaming=True,
        )
        self.stream_handler = self.stream_handler

    def _adapt_messages(self, convo: Convo) -> list[dict[str, str]]:
        """
        Adapt the conversation messages to the format expected by the GigaChat Claude model.

        Claude only recognizes "user" and "assistant" roles, and requires them to be switched
        for each message (ie. no consecutive messages from the same role).

        :param convo: Conversation to adapt.
        :return: Adapted conversation messages.
        """
        messages = []
        for msg in convo.messages:
            if msg["role"] == "function":
                raise ValueError("GigaChat Claude doesn't support function calling")

            role = "user" if msg["role"] in ["user", "system"] else "assistant"
            if messages and isinstance(messages[-1], HumanMessage) and role == "user":
                messages[-1] = HumanMessage(content=messages[-1].content + "\n\n" + msg["content"])
            elif messages and isinstance(messages[-1], SystemMessage) and role == "assistant":
                messages[-1] = SystemMessage(content=messages[-1].content + "\n\n" + msg["content"])
            elif role == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif role == "assistant":
                messages.append(SystemMessage(content=msg["content"]))

        return messages

    async def _make_request(
        self,
        convo: Convo,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> tuple[str, int, int]:
        messages = self._adapt_messages(convo)
        completion_kwargs = {
            "max_tokens": MAX_TOKENS,
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }

        if "bedrock/gigachat" in self.config.base_url:
            completion_kwargs["extra_headers"] = {"gigachat-version": "bedrock-2023-05-31"}

        if "sonnet" in self.config.model:
            if "extra_headers" in completion_kwargs:
                completion_kwargs["extra_headers"]["gigachat-beta"] = "max-tokens-3-5-sonnet-2024-07-15"
            else:
                completion_kwargs["extra_headers"] = {"gigachat-beta": "max-tokens-3-5-sonnet-2024-07-15"}
            completion_kwargs["max_tokens"] = MAX_TOKENS_SONNET

        if json_mode:
            completion_kwargs["response_format"] = {"type": "json_object"}

        response = []
        async with self.client(messages) as stream:
            async for content in stream.text_stream:
                response.append(content)
                if self.stream_handler:
                    await self.stream_handler(content)

            # TODO: get tokens from the final message
            final_message = await stream.get_final_message()
            final_message.content

        response_str = "".join(response)

        # Tell the stream handler we're done
        if self.stream_handler:
            await self.stream_handler(None)

        return response_str, final_message.usage.input_tokens, final_message.usage.output_tokens


__all__ = ["GigaChatClient"]
