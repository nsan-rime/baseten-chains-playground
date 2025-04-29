import asyncio

import truss_chains as chains

from typing import (
    Any,
    AsyncIterator
)

class PreLlama(chains.ChainletBase):
    """
    CPU Chainlet: Splits long documents/paragraphs into shorter chunks before sending to Llama.
    """
    async def run_remote(self, very_long_text: str) -> AsyncIterator[str]:
        for shorter_text in very_long_text.split("."):
            yield shorter_text.strip()

class MockLlama(chains.ChainletBase):
    """
    GPU Chainlet (H100): Mock Llama, processes sentences and streams back one token at a time.
    """
    async def run_remote(self, shorter_text: str) -> AsyncIterator[str]:
        for token in shorter_text.split(" "):
            yield token

class PostLlama(chains.ChainletBase):
    """
    GPU Chainlet (A10G): Aggregates Llama-emited tokens into longer spans and streams back spans.
    """

    def __init__(self):
        self.buffer = []

    async def run_remote(self, token: str) -> AsyncIterator[str]:
        self.buffer.append(token)

        if len(self.buffer) == 2:
            yield_str = " ".join(self.buffer)
            self.buffer = []
            yield yield_str

@chains.mark_entrypoint
class MyChainlet(chains.ChainletBase):
    def __init__(
        self,
        prellama=chains.depends(PreLlama),
        llama=chains.depends(MockLlama),
        postllama=chains.depends(PostLlama)
    ) -> None:

        self._prellama = prellama
        self._llama = llama
        self._postllama = postllama

    async def run_remote(self, very_long_text: str) -> AsyncIterator[str]:

        async for shorter_text in self._prellama.run_remote(very_long_text):

            async for token in self._llama.run_remote(shorter_text):

                async for token_span in self._postllama.run_remote(token):

                    yield token_span

if __name__ == "__main__":

    with chains.run_local():

        my_chainlet = MyChainlet()

        async def collect_chunks():
            chunks = []
            async for chunk in my_chainlet.run_remote(very_long_text="this is a test. this is another test."):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect_chunks())
        
        print(chunks)
