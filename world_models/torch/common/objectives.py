from abc import ABC, abstractmethod

from world_models.torch.common.utils import symlog_squared_error


class RepresentationObjective(ABC):
    @abstractmethod
    def compute_loss(self, context): ...

    @property
    @abstractmethod
    def requires_decoder(self): ...

    @property
    @abstractmethod
    def requires_target_encoder(self): ...

    @property
    @abstractmethod
    def requires_augmentation(self): ...


class ReconstructionObjective(RepresentationObjective):
    def __init__(self, weight=1.0, obs_type="image"):
        super().__init__()
        self.weight = weight
        self.obs_type = obs_type

    @property
    def requires_decoder(self) -> bool:
        return True

    def compute_loss(self, context):
        obs = context["obs"]
        reconstructions = context["reconstructions"]

        if self.obs_type == "image":
            loss = ((obs - reconstructions) ** 2).flatten(2).sum(-1).mean()
        else:
            loss = symlog_squared_error(obs, reconstructions)

        return loss, {"loss/reconstruction_loss": loss.item()}
