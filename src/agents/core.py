# -*- coding: utf-8 -*
import abc
import asyncio
import datetime
import time
from typing import Any, Union

import more_itertools

from ..core.functional import safe_iter


class Agent(abc.ABC):
    @abc.abstractmethod
    def generation_hook(self, prompt: str) -> Any:
        pass

    def pre_generation_hook(self, prompt: str) -> str:
        return prompt

    def post_generation_hook(self, output: Any) -> dict[str, Any]:
        return dict(output=output)

    def generate_one(self, prompt: str) -> dict[str, Any]:
        return self.post_generation_hook(
            self.generation_hook(self.pre_generation_hook(prompt))
        ) | dict(prompt=prompt, timestamp=datetime.datetime.now())

    async def generate_async(self, prompts: list[str]) -> list[dict[str, Any]]:
        ret = []
        loop = asyncio.get_running_loop()
        for cdx, chunk in enumerate(more_itertools.chunked(prompts, n=60)):
            # Trying to respect rate limits.
            if cdx > 0:
                time.sleep((60 / 5000) * len(chunk) * 10)
            # Generate the chunk.
            ret += await asyncio.gather(
                *[
                    loop.run_in_executor(None, self.generate_one, prompt)
                    for prompt in chunk
                ]
            )
        return ret

    def generate(self, prompts: Union[str, list[str]]) -> list[dict[str, Any]]:
        return asyncio.run(self.generate_async(list(safe_iter(prompts))))
