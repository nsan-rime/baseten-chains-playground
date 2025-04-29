import truss_chains as chains

from typing import (
    Any,
    AsyncIterator,
    Dict
)

# Find .rv-root file to set location of root directory
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False, indicator=[".rv-root"])

from preprocessing.pipelines import ArcanaEnglishPipeline

class DeployedArcana(chains.StubBase):
    async def run_remote(self, text: str, speaker: str) -> AsyncIterator[str]:
        async for chunk in await self.predict_async_stream(inputs={"text": text, "speaker": speaker}):
            yield chunk

@chains.mark_entrypoint
class ChunkedArcana(chains.ChainletBase):
    # `remote_config` defines the resources required for this chainlet.
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements=[
                "nemo-text-processing==1.1.0",
                "spacy==3.8.5",
                "rootutils==1.0.7"
            ]
        )
    )

    def __init__(
        self,
        context: chains.DeploymentContext = chains.depends_context()
    ) -> None:

        self._preprocess = ArcanaEnglishPipeline()

        self._arcana = DeployedArcana.from_url(
            "https://model-zq8d5rdw.api.baseten.co/environments/production/predict",
            context,
            options=chains.RPCOptions(retries=3),
        )

    async def run_remote(self, text: str, speaker: str) -> AsyncIterator[str]:

        header_yielded = False

        preprocessed_texts, preprocessing_artifacts = self._preprocess(text, verbose=True)

        for i, text_chunk in enumerate(preprocessed_texts):
            
            print(f"Generating: {i}")

            async for audio_bytes in self._arcana.run_remote(text_chunk, speaker):
                # Yield header only once (at the start of the first text chunk)
                # otherwise mid-audio header is played back as a popping sound
                # should be able to remove this once we can request headerless audio/pcm
                # to arcana endpoint 
                if audio_bytes.startswith(b'RIFF'):
                    if not header_yielded:
                        yield audio_bytes
                        header_yielded = True
                else:
                    yield audio_bytes

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

        text = "His mother had always taught him not to ever think of himself as better than others. He'd tried to live by this motto. He never looked down on those who were less fortunate or who had less money than him. But the stupidity of the group of people he was talking to made him change his mind. The choice was red, green, or blue. It didn't seem like an important choice when he was making it, but it was a choice nonetheless. Had he known the consequences at that time, he would likely have considered the choice a bit longer. In the end, he didn't and ended up choosing blue. I'm so confused by your ridiculous meltdown that I must insist on some sort of explanation for your behavior towards me. It just doesn't make any sense. There's no way that I deserved the treatment you gave me without an explanation or an apology for how out of line you have been. The box sat on the desk next to the computer. It had arrived earlier in the day and business had interrupted her opening it earlier. She didn't who had sent it and briefly wondered who it might have been. As she began to unwrap it, she had no idea that opening it would completely change her life. It was cloudy outside but not really raining. There was a light sprinkle at most and there certainly wasn't a need for an umbrella. This hadn't stopped Sarah from pulling her umbrella out and opening it. It had nothing to do with the weather or the potential rain later that day. Sarah used the umbrella to hide. It probably seemed trivial to most people, but it mattered to Tracey. She wasn't sure why it mattered so much to her, but she understood deep within her being that it mattered to her. So for the 365th day in a row, Tracey sat down to eat pancakes for breakfast. The piano sat silently in the corner of the room. Nobody could remember the last time it had been played. The little girl walked up to it and hit a few of the keys. The sound of the piano rang throughout the house for the first time in years. In the upstairs room, confined to her bed, the owner of the house had tears in her eyes. Do you think you're living an ordinary life? You are so mistaken it's difficult to even explain. The mere fact that you exist makes you extraordinary. The odds of you existing are less than winning the lottery, but here you are. Are you going to let this extraordinary opportunity pass? There was a time in his life when her rudeness would have set him over the edge. He would have raised his voice and demanded to speak to the manager. That was no longer the case. He barely reacted at all, letting the rudeness melt away without saying a word back to her. He had been around long enough to know where rudeness came from and how unhappy the person must be to act in that way. All he could do was feel pity and be happy that he didn't feel the way she did to lash out like that. The words hadn't flowed from his fingers for the past few weeks. He never imagined he'd find himself with writer's block, but here he sat with a blank screen in front of him. That blank screen taunting him day after day had started to play with his mind. He didn't understand why he couldn't even type a single word, just one to begin the process and build from there. And yet, he already knew that the eight hours he was prepared to sit in front of his computer today would end with the screen remaining blank."
        speaker = "luna"

        async def collect_chunks():
            chunks = []
            async for chunk in chunked_arcana.run_remote(text=text, speaker=speaker):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect_chunks())

        audio_samples = np.concatenate([ np.frombuffer(c, dtype=np.int16) for (i, c) in enumerate(chunks) if i > 0 ])

        sf.write("/workspace/tmp/samples.wav", audio_samples, 24_000)

        