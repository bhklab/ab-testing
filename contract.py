from pydantic import BaseModel
from input import SUPPORTED_INPUT_TYPES
from output import SUPPORTED_OUTPUT_TYPES


class ModelContract(BaseModel):

    model_name: str
    input_types: list[SUPPORTED_INPUT_TYPES]
    output_type: SUPPORTED_OUTPUT_TYPES


