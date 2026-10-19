"""Boltz-2 binding-affinity constraint for scoring small-molecule ligands against a protein target."""

from typing import Literal

from numpy import clip
from proto_tools import (
    Boltz2AffinityConfig,
    Boltz2AffinityInput,
    Complex,
    run_boltz2_affinity,
)

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils.base import BaseConfig, ConfigField


class Boltz2AffinityConstraintConfig(BaseConfig):
    """Configuration for the Boltz-2 affinity constraint.

    Scores a small-molecule ligand bound to a protein target with Boltz-2's affinity head,
    which predicts a binding affinity (``affinity_pred_value``, log10 IC50 in μM, lower is
    stronger binding) and a binder probability (``affinity_probability_binary``, 0-1, higher
    is more likely a binder). Both are converted into a penalty in ``[0.0, 1.0]`` (0 = best)
    so the constraint can rank candidates in a generative screen, where a lower energy is better.

    Each complex passed to the constraint must contain at least one protein chain (the target)
    and exactly one ligand chain (the binder); Boltz-2 auto-detects the sole ligand.

    Attributes:
        return_component (Literal['binding_penalty', 'affinity_value', 'affinity_probability']):
            Which penalty to return. ``'binding_penalty'`` (default) is the weighted combination
            of the value and probability penalties; ``'affinity_value'`` uses only the predicted
            affinity; ``'affinity_probability'`` uses only the binder probability.
        desired_affinity_value (float): Target predicted affinity (log10 IC50 μM); the value
            penalty is 0 at or below this and rises to 1 one tolerance above it. Default ``-2.0``
            (≈10 nM IC50).
        affinity_value_tolerance (float): Span in log10 IC50 units over which the value penalty
            ramps from 0 to 1 above the target.
        probability_weight (float): Weight on the probability penalty in ``'binding_penalty'``.
        value_weight (float): Weight on the value penalty in ``'binding_penalty'``.
        boltz2_affinity_config (Boltz2AffinityConfig): Advanced Boltz-2 affinity configuration
            (structure and affinity passes, MSA usage, device, verbosity). The complexes to
            score are derived from the input sequences and passed via ``Boltz2AffinityInput``.
    """

    return_component: Literal["binding_penalty", "affinity_value", "affinity_probability"] = ConfigField(
        default="binding_penalty",
        title="Return Component",
        description="Penalty to return: combined 'binding_penalty', 'affinity_value', or 'affinity_probability'.",
    )
    desired_affinity_value: float = ConfigField(
        default=-2.0,
        title="Desired Affinity Value",
        description="Target predicted affinity (log10 IC50 μM); penalty is 0 at or below this value.",
    )
    affinity_value_tolerance: float = ConfigField(
        default=4.0,
        gt=0.0,
        title="Affinity Value Tolerance",
        description="log10 IC50 span above target over which the value penalty ramps from 0 to 1.",
    )
    probability_weight: float = ConfigField(
        default=0.5,
        ge=0.0,
        title="Probability Weight",
        description="Weight on the binder-probability penalty in the combined binding_penalty.",
    )
    value_weight: float = ConfigField(
        default=0.5,
        ge=0.0,
        title="Value Weight",
        description="Weight on the affinity-value penalty in the combined binding_penalty.",
    )
    boltz2_affinity_config: Boltz2AffinityConfig = ConfigField(
        default_factory=Boltz2AffinityConfig,
        title="Boltz2 Affinity Config",
        description="Boltz-2 affinity configuration for structure and affinity prediction.",
    )


@constraint(
    key="boltz2-affinity",
    label="Boltz2 Affinity",
    config=Boltz2AffinityConstraintConfig,
    description="Predicted binding affinity (log10 IC50) and binder probability of a ligand to a protein target via Boltz-2",
    uses_gpu=True,
    tools_called=["boltz2-affinity"],
    category="protein_structure",
    supported_sequence_types=["protein", "ligand"],
    input_labels=None,
)
def boltz2_affinity_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: Boltz2AffinityConstraintConfig
) -> list[ConstraintOutput]:
    """Score ligand-protein binding affinity with Boltz-2.

    Builds a protein-ligand complex from each input tuple, predicts binding affinity with
    Boltz-2, and converts the predicted affinity and binder probability into a penalty in
    ``[0.0, 1.0]`` (0 = strongest predicted binding). Each complex must contain at least one
    protein chain and exactly one ligand chain.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Complexes to score, each a tuple of
            ``Sequence`` objects (protein target chains plus one ligand SMILES).
        config (Boltz2AffinityConstraintConfig): Target affinity, tolerance, penalty weights,
            and Boltz-2 affinity parameters.

    Returns:
        list[ConstraintOutput]: Per-complex penalty in ``[0.0, 1.0]`` (0 = best). The predicted
            pose is attached to the first slot of each complex. ``metadata`` carries
            ``boltz2_affinity`` (a one-element list of dicts with ``penalty``, the raw ``metrics``,
            and the component ``penalties``).

    Raises:
        ValueError: If Boltz-2 does not return the affinity metrics required by the selected
            ``return_component``.
    """
    boltz_complexes = [
        Complex(chains=[{"sequence": s.sequence, "entity_type": s.sequence_type} for s in sequence_tuple])
        for sequence_tuple in input_sequences
    ]

    inputs = Boltz2AffinityInput(complexes=boltz_complexes)
    outputs = run_boltz2_affinity(inputs=inputs, config=config.boltz2_affinity_config)

    results: list[ConstraintOutput] = []
    for seq_obj_tuple, structure in zip(input_sequences, outputs.structures, strict=True):
        metrics = structure.metrics
        penalties: dict[str, float] = {}

        prob = metrics.get("affinity_probability_binary")
        if prob is not None:
            penalties["affinity_probability"] = float(clip(1.0 - float(prob), 0.0, 1.0))

        value = metrics.get("affinity_pred_value")
        if value is not None:
            deviation = (float(value) - config.desired_affinity_value) / max(config.affinity_value_tolerance, 1e-9)
            penalties["affinity_value"] = float(clip(deviation, 0.0, 1.0))

        penalty = _select_penalty(penalties, config)

        n = len(seq_obj_tuple)
        results.append(
            ConstraintOutput(
                score=penalty,
                metadata={
                    "boltz2_affinity": [{"penalty": penalty, "metrics": metrics, "penalties": penalties}]
                },
                structures=(structure,) + (None,) * (n - 1),
            )
        )

    return results


def _select_penalty(penalties: dict[str, float], config: Boltz2AffinityConstraintConfig) -> float:
    """Combine or select the affinity penalties for the configured ``return_component``."""
    if config.return_component == "affinity_probability":
        if "affinity_probability" not in penalties:
            raise ValueError("Boltz-2 did not return 'affinity_probability_binary' for the requested component.")
        return penalties["affinity_probability"]
    if config.return_component == "affinity_value":
        if "affinity_value" not in penalties:
            raise ValueError("Boltz-2 did not return 'affinity_pred_value' for the requested component.")
        return penalties["affinity_value"]

    weights = {
        "affinity_probability": config.probability_weight,
        "affinity_value": config.value_weight,
    }
    used = {name: weights[name] for name in penalties if weights.get(name, 0.0) > 0.0}
    if not used:
        raise ValueError(
            "Boltz-2 returned no affinity metrics with a positive weight; "
            "set probability_weight/value_weight or check the affinity output."
        )
    total = sum(used.values()) or 1.0
    return float(clip(sum((w / total) * penalties[name] for name, w in used.items()), 0.0, 1.0))
