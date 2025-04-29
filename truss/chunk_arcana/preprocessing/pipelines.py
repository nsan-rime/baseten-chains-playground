# Find .rv-root file to set location of root directory
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False, indicator=[".rv-root"])

from preprocessing.processor_text_chunk import EnglishTextChunkProcessor

from preprocessing import RimeProcessorPipeline

class ArcanaEnglishPipeline(RimeProcessorPipeline):

    def __init__(self):
        super().__init__(
            text_chunk = EnglishTextChunkProcessor(max_nchars=500)
        )
