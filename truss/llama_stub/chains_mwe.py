import asyncio
import os

import truss_chains as chains

from typing import (
    Any,
    AsyncIterator
)

# Snippet from DeployedWhisper example: https://docs.baseten.co/reference/sdk/chains#class-truss-chains-stubbase
class DeployedLlama(chains.StubBase):
    async def run_remote(self, text: str) -> AsyncIterator[str]:
        async for chunk in await self.predict_async_stream(inputs={"text": text}):
            yield chunk.decode()

@chains.mark_entrypoint
class MyChainlet(chains.ChainletBase):

    # Snippet from DeployedWhisper example: https://docs.baseten.co/reference/sdk/chains#class-truss-chains-stubbase
    def __init__(
        self,
        context: chains.DeploymentContext = chains.depends_context()
    ) -> None:

        self._llama = DeployedLlama.from_url(
            "https://model-8w6z6553.api.baseten.co/development/predict",
            context,
            options=chains.RPCOptions(retries=3),
        )

    # Snippet from: https://docs.baseten.co/development/chain/streaming#low-level-streaming
    async def run_remote(self, text: str) -> AsyncIterator[str]:
        async for text_chunk in self._llama.run_remote(text):
            yield text_chunk

if __name__ == "__main__":

    with chains.run_local(
        secrets={"baseten_chain_api_key": os.environ["BASETEN_API_KEY"]}
    ):

        my_chainlet = MyChainlet()

        async def collect_chunks():
            chunks = []
            async for chunk in my_chainlet.run_remote(text="this is a test"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect_chunks())
        
        print(chunks)
