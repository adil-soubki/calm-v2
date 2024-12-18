# -*- coding: utf-8 -*
import logging
from typing import Any, Optional, Union

import torch
import transformers as tf

from ..agents.core import Agent
from ..core.functional import safe_iter


class HFAgent(Agent):
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        **generation_kwargs: Any
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.generation_kwargs = generation_kwargs

        # XXX: This should not be hardcoded.
        self.pipeline = tf.pipeline(
            "text2text-generation",
            model=self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            model_kwargs=dict(low_cpu_mem_usage=True)
        )
        # Only return new tokens.
        if self.pipeline.task == "text-generation":
            self.generation_kwargs["return_full_text"] = False
        # Warn if ignoring the system prompt.
        if (
            self.pipeline.tokenizer.chat_template is None and 
            self.system_prompt is not None
        ):
            logging.getLogger(__name__).warn(
                f"{self.model_name} has no chat template but a system prompt "
                f"was provided. This system prompt will be ignored."
            )

    def generation_hook(self, prompt: str) -> Any:
        if self.pipeline.tokenizer.chat_template is None:
            return self.pipeline(prompt, **self.generation_kwargs)[0]["generated_text"]
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.pipeline(messages, **self.generation_kwargs)[0]["generated_text"]

    def post_generation_hook(self, output: Any) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "generation": output,
            **self.generation_kwargs,
            **({"system_prompt": self.system_prompt} if self.system_prompt else {}),
        }

    # XXX: This should be set up to allow batching.
    def generate(self, prompts: Union[str, list[str]]) -> list[dict[str, Any]]:
        return [self.generate_one(p) for p in safe_iter(prompts)]
