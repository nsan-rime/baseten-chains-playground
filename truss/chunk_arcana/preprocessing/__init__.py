import time

from functools import reduce

class RimeProcessorWrapper:
    def __init__(self, disabled=False):
        self.default_state=disabled
        self.disabled=self.default_state

    def preprocess(self, x, _artifacts={}):
        return x, _artifacts

    def process(self, x, _artifacts={}):
        return x, _artifacts

    def postprocess(self, x, _artifacts={}):
        return x, _artifacts

    def __call__(self, x, _artifacts={}):
        if not self.disabled:
            x, _artifacts = self.preprocess(x, _artifacts)
            x, _artifacts = self.process(x, _artifacts)
            x, _artifacts = self.postprocess(x, _artifacts)
        return x, _artifacts

class RimeProcessorPipeline:
    def __init__(self, **kwargs):
        self.pipeline = []

        if len(kwargs) == 0:
            ValueError("Pipeline is empty!")

        for name, callable in kwargs.items():
            setattr(self, name, callable)
            self.pipeline.append(getattr(self, name))

    def reset(self):
        for p in self.pipeline:
            p.disabled = p.default_state

    def set_disabled(self, **kwargs):
        for name, flag in kwargs.items():
            _proc = getattr(self, name)
            _proc.disabled = flag

    def __call__(self, input, verbose=False):
        # Given self.pipeline sequence of processes to be applied [ p0, p1, ..., pn ]
        # Apply to input, where output = pn(p...(p1(p0(input))))
        if verbose:
            pipe_start = time.perf_counter()
        
        output = []
        for i, p in enumerate(self.pipeline):
            _input = input if i == 0 else output[i - 1]['output']

            _input_text = _input[0] if type(_input) is list else _input
            _input_artifacts = _input[1] if type(_input) is list else None

            if verbose:
                proc_start = time.time_ns()

            _output_text, _output_artifacts = p(*_input) if type(_input) is list else p(_input, {})

            if verbose:
                proc_end = time.time_ns()
                proc_dur_micro_sec = (proc_end - proc_start) / 1000

                print(f"{i}: {type(p).__name__} (Disabled: {p.disabled}) ({proc_dur_micro_sec:.0f} μs)")
                print(f"\tInput:\t\t{_input_text}")
                print(f"\tOutput:\t\t{_output_text}")
                print(f"\tArtifacts:\t{_output_artifacts}\n")

            output.append({
                'processor' : type(p).__name__,
                'input' : _input,
                'output' : [_output_text, _output_artifacts]
            })

        if verbose:
            pipe_end = time.perf_counter()
            pipe_dur_ms = (pipe_end - pipe_start) * 1000
            print(f"Pipeline took {pipe_dur_ms:.0f} ms")

        return output[-1]['output']
