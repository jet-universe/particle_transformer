import torch
from weaver.nn.model.LGATr import LGATrTagger
from weaver.utils.logger import _logger


class LGATrTaggerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = LGATrTagger(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "mod.cls_token",
        }

    def forward(self, points, features, lorentz_vectors, mask):
        f = self.mod(features, v=lorentz_vectors, mask=mask)
        return f


def get_model(data_config, **kwargs):
    use_fully_connected_subgroup = False
    if not use_fully_connected_subgroup:
        import weaver.nn.model.gatr.primitives.linear

        weaver.nn.model.gatr.primitives.linear.USE_FULLY_CONNECTED_SUBGROUP = False

    cfg = dict(
        in_s_channels=len(data_config.input_dicts["pf_features"]),
        num_classes=len(data_config.label_value),
        # symmetry-breaking configurations
        spurion_token=True,
        beam_reference="xyplane",
        add_time_reference=True,
        two_beams=True,
        # network configurations
        hidden_mv_channels=16,
        hidden_s_channels=32,
        num_blocks=12,
        num_heads=8,
        double_layernorm=True,
        head_scale=True,
    )

    cfg.update(**kwargs)
    _logger.info("Model config: %s" % str(cfg))

    model = LGATrTaggerWrapper(**cfg)

    model_info = {
        "input_names": list(data_config.input_names),
        "input_shapes": {
            k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()
        },
        "output_names": ["softmax"],
        "dynamic_axes": {
            **{k: {0: "N", 2: "n_" + k.split("_")[0]} for k in data_config.input_names},
            **{"softmax": {0: "N"}},
        },
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
