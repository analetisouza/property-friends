class EmptyPath(Exception):
    """
    Is raised when the path provided is an empty string or None.
    """
    def __init__(self, attribute):
        self.message = f"The {attribute} parameter provided is empty."
        super().__init__(self.message)


class DataNotLoaded(Exception):
    """
    Is raised when the class Loader is not initialized before training.
    """
    def __init__(self):
        self.message = "The data was not loaded. Please initialize the class Loader before training."
        super().__init__(self.message)


class ModelNotTrained(Exception):
    """
    Is raised when the class Trainer is not initialized before training.
    """
    def __init__(self):
        self.message = "The model was not trained. Please initialize the class Trainer before evaluating."
        super().__init__(self.message)
