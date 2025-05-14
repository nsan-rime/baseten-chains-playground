import time
import uuid

import pydantic
import truss_chains as chains

from typing import (
    Any,
    List,
    AsyncIterator
)

class vLLMSamplingParams(pydantic.BaseModel):
    temperature: float = 0.1
    top_p: float = 0.75
    max_tokens: int = 5000
    stop_token_ids: List[ int ] = [ 128_258 ]
    repetition_penalty: float = 1.25

class TTSRequest(pydantic.BaseModel):
    text: str
    request_id: str
    speaker: type(None) = None
    sampling_params: vLLMSamplingParams

class TTSDeployment(chains.StubBase):
    async def run_remote(self, request: TTSRequest) -> AsyncIterator[str]:
        async for chunk in await self.predict_async_stream(inputs=request):
            yield chunk

@chains.mark_entrypoint
class BenchmarkEntrypoint(chains.ChainletBase):
    def __init__(
        self,
        context: chains.DeploymentContext = chains.depends_context()
    ) -> None:

        predict_url = "https://model-zq8d5rdw.api.baseten.co/environments/production/predict"
        self._tts = TTSDeployment.from_url(predict_url, context, options=chains.RPCOptions(retries=3))

    async def run_remote(self, text: str = "", speaker: str = None) -> AsyncIterator[str]:
        request = TTSRequest(
            text = text,
            request_id = "benchmark_" + str(uuid.uuid4().hex),
            sampling_params = vLLMSamplingParams()
        )

        first_chunk_yielded = False
        start = time.time()

        async for audio_chunk in self._tts.run_remote(request):
            if not first_chunk_yielded:
                print(f"{request.request_id}, TTFB: {(time.time() - start) * 1000} ms")
                first_chunk_yielded = True

            yield audio_chunk

if __name__ == "__main__":

    # Local dev

    import asyncio
    import os

    with chains.run_local(
        secrets={"baseten_chain_api_key": os.environ["BASETEN_API_KEY"]}
    ):

        e = BenchmarkEntrypoint()
            
        async def collect_chunks():
            audio_bytes = b''
            async for chunk in e.run_remote(text="This is a test."):
                if chunk:
                    audio_bytes += chunk
            return audio_bytes

        audio_bytes = asyncio.run(collect_chunks())
