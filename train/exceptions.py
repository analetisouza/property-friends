class InvalidPath(Exception):
    def __init__(self, attribute):
        self.message = "The {} parameter provided is invalid.".format(attribute)
        super().__init__(self.message)
