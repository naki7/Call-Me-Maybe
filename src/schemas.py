from typing import Literal

from pydantic import BaseModel, ConfigDict


ParameterType = Literal[
    "number",
    "string",
    "boolean",
    "list",
    "dictionary",
]


class ParameterDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: ParameterType


class ReturnDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: ParameterType


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str
    name: str
    parameters: dict[str, object]
