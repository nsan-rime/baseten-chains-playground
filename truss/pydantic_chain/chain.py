import asyncio

import truss_chains as chains

from pydantic import (
    BaseModel,
    Field,
    constr,
    model_validator,
    PositiveFloat
)
from typing import Literal

NonEmptyStr = constr(strip_whitespace=True, min_length=1)

# Pydanctic model to be used by Baseten server for validation
# even before run_remote is run
class UserPayload(BaseModel):
    # Required input params
    text: NonEmptyStr
    speaker: NonEmptyStr
    
    # Runtime params:
    rimeLogging: bool = False
    
    # Preprocessing params
    noTextNormalization: bool = False
    
    # vLLM params
    temperature: float = Field(gt=0, le=1, default=0.1)
    top_p: float = Field(gt=0, le=1, default=0.75)
    repetition_penalty: PositiveFloat = 1.25
    
    # Output params
    audioFormat: Literal['wav', 'pcm', 'mp3', 'mulaw'] = 'wav'
    samplingRate: Literal[ 8_000, 16_000, 22_050, 44_100, 48_000 ] = 24_000

    @model_validator(mode='after')
    def mulaw_8khz(self):
        if self.audioFormat == "mulaw":
            self.samplingRate = 8_000
        return self

# Payload used from client on Baseten playground
client_payload = {
    "payload" : {
        "text" : "this is a test.",
        "speaker" : "luna"
    }
}

@chains.mark_entrypoint
class HelloWorld(chains.ChainletBase):
    async def run_remote(self, payload: UserPayload) -> str:
        payload_str = payload.model_dump_json(indent=4)
        print(payload_str)
        return payload_str

# Test the Chain locally
if __name__ == "__main__":
    with chains.run_local():

        local_payload = UserPayload(**client_payload["payload"])

        e = HelloWorld()
        result = asyncio.get_event_loop().run_until_complete(
            e.run_remote(local_payload)
        )
