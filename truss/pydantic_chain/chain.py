import asyncio

import random
import truss_chains as chains

from pydantic import (
    BaseModel,
    PositiveInt,
)

class RandInt(chains.ChainletBase):
    async def run_remote(self, max_value: int) -> int:
        return random.randint(1, max_value)

class HelloWordRequest(BaseModel):
    max_value: PositiveInt

@chains.mark_entrypoint
class HelloWorld(chains.ChainletBase):
    def __init__(self, rand_int=chains.depends(RandInt, retries=3)) -> None:
        self._rand_int = rand_int

    async def run_remote(self, payload: HelloWordRequest) -> str:
        num_repetitions = await self._rand_int.run_remote(payload.max_value)
        return "Hello World! " * num_repetitions

# Test the Chain locally
if __name__ == "__main__":
    with chains.run_local():
        payload = HelloWordRequest(max_value=5)

        hello_chain = HelloWorld()
        result = asyncio.get_event_loop().run_until_complete(
            hello_chain.run_remote(payload)
        )
        print(result)
