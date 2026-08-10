from pydantic import BaseModel
from ab_testing.input import SUPPORTED_INPUT_TYPES
from ab_testing.output import SUPPORTED_OUTPUT_TYPES


class ModelContract(BaseModel):

    model_name: str
    input_types: list[SUPPORTED_INPUT_TYPES]
    output_type: SUPPORTED_OUTPUT_TYPES


