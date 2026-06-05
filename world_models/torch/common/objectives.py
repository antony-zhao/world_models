from abc import ABC, abstractmethod
from typing import List

from world_models.torch.common.utils import symlog_squared_error


class RepresentationObjective(ABC):
    @abstractmethod
    def __call__(self, context): ...

    @property
    def requires_decoder(self):
        return False

    @property
    def requires_target_encoder(self):
        return False

    @property
    def requires_augmentation(self):
        return False


class CompoundObjective(RepresentationObjective):
    def __init__(self, objectives: List):
        super().__init__()
        self.objectives = objectives

    @property
    def requires_decoder(self):
        return any(o.requires_decoder for o in self.objectives)

    @property
    def requires_target_encoder(self):
        return any(o.requires_target_encoder for o in self.objectives)

    @property
    def requires_augmentation(self):
        return any(o.requires_augmentation for o in self.objectives)

    def __call__(self, context):
        total_loss = 0.0
        merged_dict = {}
        for obj in self.objectives:
            loss, metrics = obj(context)
            total_loss = total_loss + loss
            merged_dict |= metrics
        return total_loss, merged_dict


class ReconstructionObjective(RepresentationObjective):
    def __init__(self, weight=1.0, obs_type="image"):
        super().__init__()
        self.weight = weight
        self.obs_type = obs_type

    @property
    def requires_decoder(self) -> bool:
        return True

    def __call__(self, context):
        obs = context["obs"]
        reconstructions = context["reconstructions"]

        if self.obs_type == "image":
            loss = ((obs - reconstructions) ** 2).flatten(2).sum(-1).mean()
        else:
            loss = symlog_squared_error(obs, reconstructions)

        return self.weight * loss, {"loss/reconstruction_loss": loss.item()}
