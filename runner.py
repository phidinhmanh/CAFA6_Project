# runner.py
class PipelineRunner:
    def __init__(self, config, context=None):
        self.config = config
        self.context = context or {}

    def add(self, step):
        self.context = step.run(self.context, self.config)
        return self

    def run(self, steps):
        for step in steps:
            self.add(step)
        return self.context
