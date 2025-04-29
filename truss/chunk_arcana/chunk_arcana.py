import truss_chains as chains

from typing import (
    Any,
    AsyncIterator,
    Dict
)

class DeployedArcana(chains.StubBase):
    async def run_remote(self, text: str, speaker: str) -> AsyncIterator[str]:
        async for chunk in await self.predict_async_stream(inputs={"text": text, "speaker": speaker}):
            yield chunk

@chains.mark_entrypoint
class ChunkedArcana(chains.ChainletBase):

    def __init__(
        self,
        context: chains.DeploymentContext = chains.depends_context()
    ) -> None:

        self._arcana = DeployedArcana.from_url(
            "https://model-zq8d5rdw.api.baseten.co/environments/production/predict",
            context,
            options=chains.RPCOptions(retries=3),
        )

    async def run_remote(self, text: str, speaker: str) -> AsyncIterator[str]:
        # Naive split on periods for dev work
        sentences = [ s if s.endswith('.') else s+'.' for s in text.split(". ") ]

        for s in sentences:
            async for audio_chunk in self._arcana.run_remote(s, speaker):
                yield audio_chunk

if __name__ == "__main__":
    # Import dev dependencies here (not required on Baseten image)
    import asyncio
    import os

    import numpy as np
    import soundfile as sf

    with chains.run_local(
        secrets={"baseten_chain_api_key": os.environ["BASETEN_API_KEY"]}
    ):

        chunked_arcana = ChunkedArcana()

        text = "this is a test. this is another test."
        speaker = "luna"

        async def collect_chunks():
            chunks = []
            async for chunk in chunked_arcana.run_remote(text=text, speaker=speaker):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect_chunks())

        with open("/workspace/tmp/two-sentence-chunked-arcana-audio.wav", "wb") as f:
            for chunk in chunks:
                if chunk:
                    f.write(chunk)
