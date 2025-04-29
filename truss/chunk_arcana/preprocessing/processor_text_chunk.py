# Find .rv-root file to set location of root directory
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False, indicator=[".rv-root"])

from preprocessing import RimeProcessorWrapper

class EnglishTextChunkProcessor(RimeProcessorWrapper):

    def __init__(self, max_nchars):
        super().__init__()

        self.max_nchars=max_nchars

        import spacy

        if not spacy.util.is_package("en_core_web_sm"):
            spacy.cli.download("en_core_web_sm")

        self.nlp = spacy.load('en_core_web_sm', enable=('tok2vec',))
        self.nlp.disable_pipe('tok2vec') # not actually required, but easier than disabling everything
        self.nlp.enable_pipe('senter')

    def process(self, x, _artifacts):

        if len(x) < self.max_nchars:
            # Return a list of 1 text if text isn't long enough to need chunking
            return [x], _artifacts

        else:
            sentences = self.nlp(x).sents

            text_chunks = []
            current_chunk = ''

            # TODO: Figure out edge cases where s is longer than max_nchars for some reason
            for s in sentences:
                potential_chunk = f"{s}" if current_chunk == '' else f"{current_chunk} {s}"

                if len(potential_chunk) < self.max_nchars:
                    current_chunk = potential_chunk
                else:
                    text_chunks.append(current_chunk)
                    current_chunk = str(s)

            text_chunks.append(current_chunk)

            return text_chunks, _artifacts

if __name__ == "__main__":

    p = EnglishTextChunkProcessor(50)

    print(p.process("This is a sentence. This is another sentence.", None)[0])
