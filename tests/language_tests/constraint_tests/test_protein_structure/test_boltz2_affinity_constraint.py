"""Tests for the Boltz-2 affinity constraint."""

from unittest.mock import patch

import pytest
from proto_tools import BFactorType, StructurePredictionOutput

from proto_language.constraint import ConstraintRegistry
from proto_language.constraint.protein_structure.boltz2_affinity_constraint import (
    Boltz2AffinityConstraintConfig,
    _select_penalty,
    boltz2_affinity_constraint,
)
from proto_language.core import Sequence
from tests.helpers.mock_structure import MockStructure

MODULE = "proto_language.constraint.protein_structure.boltz2_affinity_constraint"

PROTEIN = "MKTAYIAKQRQISFVK"
LIGAND_SMILES = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"


def _mock_output(affinity_pred_value: float, affinity_probability_binary: float) -> StructurePredictionOutput:
    """Build a mocked Boltz-2 affinity output with one scored complex."""
    structure = MockStructure(
        structure_format="cif",
        b_factor_type=BFactorType.PLDDT,
        source="boltz2-affinity",
        metrics={
            "affinity_pred_value": affinity_pred_value,
            "affinity_probability_binary": affinity_probability_binary,
        },
    )
    return StructurePredictionOutput(
        tool_id="boltz2-affinity",
        execution_time=0.0,
        success=True,
        structures=[structure],
        warnings=[],
        metadata={},
    )


def _complex() -> list[tuple[Sequence, ...]]:
    """One protein-ligand complex as the constraint receives it."""
    return [(Sequence(sequence=PROTEIN, sequence_type="protein"), Sequence(sequence=LIGAND_SMILES, sequence_type="ligand"))]


def test_combined_penalty_blends_value_and_probability():
    """binding_penalty is the weighted blend of the value and probability penalties."""
    config = Boltz2AffinityConstraintConfig()  # desired=-2.0, tol=4.0, weights 0.5/0.5
    # value penalty: (0.0 - (-2.0)) / 4.0 = 0.5 ; probability penalty: 1 - 0.8 = 0.2
    output = _mock_output(affinity_pred_value=0.0, affinity_probability_binary=0.8)
    with patch(f"{MODULE}.run_boltz2_affinity", return_value=output):
        results = boltz2_affinity_constraint(_complex(), config)

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.35)
    # predicted pose attached to the first (protein) slot, None for the ligand slot.
    assert results[0].structures[0] is not None
    assert results[0].structures[1] is None
    assert "boltz2_affinity" in results[0].metadata


def test_return_component_selects_single_metric():
    """return_component restricts the penalty to one metric."""
    output = _mock_output(affinity_pred_value=0.0, affinity_probability_binary=0.8)
    with patch(f"{MODULE}.run_boltz2_affinity", return_value=output):
        prob_only = boltz2_affinity_constraint(
            _complex(), Boltz2AffinityConstraintConfig(return_component="affinity_probability")
        )
        value_only = boltz2_affinity_constraint(
            _complex(), Boltz2AffinityConstraintConfig(return_component="affinity_value")
        )
    assert prob_only[0].score == pytest.approx(0.2)
    assert value_only[0].score == pytest.approx(0.5)


def test_strong_binder_scores_low():
    """A strongly-predicted binder (low IC50, high probability) yields a near-zero penalty."""
    output = _mock_output(affinity_pred_value=-3.0, affinity_probability_binary=1.0)
    with patch(f"{MODULE}.run_boltz2_affinity", return_value=output):
        results = boltz2_affinity_constraint(_complex(), Boltz2AffinityConstraintConfig())
    assert results[0].score == pytest.approx(0.0)


def test_select_penalty_combines_and_clips():
    """_select_penalty blends by weight for the combined component."""
    config = Boltz2AffinityConstraintConfig(probability_weight=3.0, value_weight=1.0)
    penalties = {"affinity_probability": 0.4, "affinity_value": 0.8}
    # (3*0.4 + 1*0.8) / 4 = 0.5
    assert _select_penalty(penalties, config) == pytest.approx(0.5)


def test_select_penalty_raises_when_requested_metric_missing():
    """Requesting a single component Boltz-2 did not return is an error."""
    config = Boltz2AffinityConstraintConfig(return_component="affinity_value")
    with pytest.raises(ValueError, match="affinity_pred_value"):
        _select_penalty({"affinity_probability": 0.3}, config)


def test_config_tolerance_must_be_positive():
    """A non-positive affinity tolerance is rejected."""
    with pytest.raises(ValueError):
        Boltz2AffinityConstraintConfig(affinity_value_tolerance=0.0)


def test_registered_with_protein_and_ligand_types():
    """The constraint is registered for protein and ligand inputs on GPU."""
    spec = ConstraintRegistry.get("boltz2-affinity")
    assert set(spec.supported_sequence_types) == {"protein", "ligand"}
    assert spec.uses_gpu is True
