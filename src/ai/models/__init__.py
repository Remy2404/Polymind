class ImageAttachment:
    """Represents an image attachment with base64 data."""

    def __init__(self, name: str, content_type: str, data: str):
        self.name = name
        self.content_type = content_type
        self.data = data
