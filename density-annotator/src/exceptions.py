class InputFileTypeError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class SamplingTypeError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class LoadFrameError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class PathExistError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
