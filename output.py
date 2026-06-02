from pydantic import BaseModel


class NiftiOutput(BaseModel):

    nifti_path: str


class NpzOutput(BaseModel):

    npz_path: str


SUPPORTED_OUTPUT_TYPES = [NiftiOutput, NpzOutput]