# -*- coding: utf-8 -*
from typing import Any, Optional

import backoff
import openai

from ..agents.core import Agent
from ..core import keychain


class GPTAgent(Agent):
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        **generation_kwargs: Any
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.generation_kwargs = generation_kwargs
        self.client = openai.OpenAI(api_key=keychain.get("IACS"))

    @backoff.on_exception(
        backoff.expo, (openai.RateLimitError, openai.APIConnectionError)
    )
    def generation_hook(self, prompt: str) -> Any:
        messages: list[openai.types.chat.ChatCompletionMessageParam] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.client.chat.completions.create(
            model=self.model_name, messages=messages, **self.generation_kwargs
        )

    def post_generation_hook(self, output: Any) -> dict[str, Any]:
        return {
            "model_name": output.model,
            "generation": output.choices[0].message.content,
            **self.generation_kwargs,
            **({"system_prompt": self.system_prompt} if self.system_prompt else {}),
        }
