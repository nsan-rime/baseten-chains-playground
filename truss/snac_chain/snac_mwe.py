import asyncio
import os

import truss_chains as chains

from typing import (
    Any,
    AsyncIterator
)

from snac_decode import create_wav_header, tokens_decoder

class MockLlama(chains.ChainletBase):
    async def run_remote(self) -> AsyncIterator[str]:
        for line in open("/workspace/truss/snac_chain/llama-tokens.txt"):
            yield line.strip()

@chains.mark_entrypoint
class MyChainlet(chains.ChainletBase):
    def __init__(self, llama=chains.depends(MockLlama, retries=3)) -> None:
        self._llama = llama

    async def run_remote(self) -> AsyncIterator[str]:
        yield create_wav_header()

        token_gen = self._llama.run_remote()

        async for audio_chunk in tokens_decoder(token_gen):
            yield audio_chunk

if __name__ == "__main__":

    with chains.run_local(
        secrets={"baseten_chain_api_key": os.environ["BASETEN_API_KEY"]}
    ):
        my_chainlet = MyChainlet()

    async def collect_chunks():
        chunks = []
        async for chunk in my_chainlet.run_remote():
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect_chunks())
    
    with open("/workspace/tmp/test-audio.wav", "wb") as f:
        for chunk in chunks:
            f.write(chunk)

    print("=== Saved /workspace/tmp/test-audio.wav ===")
