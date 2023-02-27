class InputFileTypeError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class SamplingTypeError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class LoadImageError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class LoadVideoError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class LoadVideoFrameError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class PathNotExistError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
