from pydantic import BaseModel, ConfigDict


class ABaseModel(BaseModel):

    model_config =  ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='forbid',
    )

class FrozenABaseModel(ABaseModel):

    model_config =  ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra='forbid',
        frozen=True,
    )
