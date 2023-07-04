class PathNotExistError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class StatsKeyNotExistError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
