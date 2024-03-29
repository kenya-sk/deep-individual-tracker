class LoadModelError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class LoadImageError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class LoadVideoError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class DatasetEmptyError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class DatasetSplitTypeError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class TensorShapeError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class DetectionSampleNumberError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class PredictionTypeError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class IndexExtractionError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
