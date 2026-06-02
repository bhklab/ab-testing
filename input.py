from tarfile import SUPPORTED_TYPES
from pydantic import BaseModel


class ImageInput(BaseModel):

    image_path: str
    image_extension: str


class TextInput(BaseModel):

    text: str


SUPPORTED_INPUT_TYPES = [ImageInput, TextInput]
