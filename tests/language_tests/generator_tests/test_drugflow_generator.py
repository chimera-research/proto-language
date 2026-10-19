"""Tests for DrugFlowGenerator (pocket-conditioned de novo ligand generation)."""

from unittest.mock import patch

import pytest
from proto_tools import (
    DrugFlowGenerateOutput,
    DrugFlowGenerationSet,
    DrugFlowMolecule,
    DrugFlowMoleculeMetrics,
)
from proto_tools.entities.structures import Structure

from proto_language.core import Segment
from proto_language.generator import DrugFlowGenerator, DrugFlowGeneratorConfig
from proto_language.generator.generator_registry import GeneratorRegistry

MODULE = "proto_language.generator.drugflow_generator"
PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"


def _config(**kwargs: object) -> DrugFlowGeneratorConfig:
    """Config with a valid target structure + reference ligand."""
    base: dict[str, object] = {"target_structure": PDB, "reference_ligand": "REF_SDF_BLOCK"}
    base.update(kwargs)
    return DrugFlowGeneratorConfig(**base)  # type: ignore[arg-type]


def _fake_output(smiles: list[str]) -> DrugFlowGenerateOutput:
    """Build a real DrugFlowGenerateOutput with one generation set of the given molecules."""
    molecules = [
        DrugFlowMolecule(
            smiles=smi,
            pose_sdf=f"POSE_{i}",
            metrics=DrugFlowMoleculeMetrics(qed=0.5, molecular_weight=120.0, num_atoms=9.0),
        )
        for i, smi in enumerate(smiles)
    ]
    return DrugFlowGenerateOutput(generation_sets=[DrugFlowGenerationSet(molecules=molecules)])


def test_init_and_contract():
    """Generator stores config and declares the ligand/de-novo contract."""
    gen = DrugFlowGenerator(_config(pocket_distance_cutoff=10.0, molecule_size=30))
    assert gen.pocket_distance_cutoff == 10.0
    assert gen.molecule_size == 30
    assert gen.generates_ligands is True
    assert gen.allows_empty_starting_sequence is True


def test_normalize_target_structure_from_pdb_content():
    """Raw PDB content is coerced to a Structure."""
    cfg = _config()
    assert isinstance(cfg.target_structure, Structure)


def test_registered_as_ligand_generator():
    """The generator is registered with the ligand sequence type and mutation category."""
    spec = GeneratorRegistry.get("drugflow")
    assert spec.supported_sequence_types == ["ligand"]
    assert spec.category == "mutation"
    assert spec.allows_empty_starting_sequence is True


def test_assign_accepts_empty_ligand_segment():
    """A de novo ligand segment (empty SMILES) accepts the generator."""
    ligand = Segment(sequence="", sequence_type="ligand", label="cand")
    gen = DrugFlowGenerator(_config())
    gen.assign(ligand)
    assert gen.is_assigned and gen.segment is ligand


@pytest.mark.parametrize("sequence_type", ["protein", "dna", "rna"])
def test_rejects_non_ligand_segment(sequence_type):
    """The generator only supports ligand segments."""
    seq = {"protein": "ACDEFG", "dna": "ACGT", "rna": "ACGU"}[sequence_type]
    gen = DrugFlowGenerator(_config())
    with pytest.raises(ValueError, match="does not support sequence type"):
        gen.assign(Segment(sequence=seq, sequence_type=sequence_type))


def test_sample_writes_smiles_pose_and_metadata():
    """_sample fills proposals with SMILES and records pose + metrics metadata."""
    ligand = Segment(sequence="", sequence_type="ligand", label="cand")
    gen = DrugFlowGenerator(_config())
    gen.assign(ligand)
    ligand.proposal_sequences = [
        type(ligand.proposal_sequences[0])(sequence="", sequence_type="ligand") for _ in range(3)
    ]

    with patch(f"{MODULE}.run_drugflow_generate", return_value=_fake_output(["CCO", "c1ccccc1", "CCN"])) as mock_run:
        gen.sample()

    assert [p.sequence for p in ligand.proposal_sequences] == ["CCO", "c1ccccc1", "CCN"]
    meta = ligand.proposal_sequences[0]._generator_metadata["drugflow"]
    assert meta["qed"] == 0.5
    assert meta["pose_sdf"] == "POSE_0"
    # The tool was asked for exactly num_proposals molecules for the one target pocket.
    assert mock_run.call_args.kwargs["config"].num_molecules_per_target == 3
    assert len(mock_run.call_args.kwargs["inputs"].proteins) == 1


def test_sample_cycles_when_tool_returns_fewer():
    """Fewer molecules than proposals are cycled to fill the pool."""
    ligand = Segment(sequence="", sequence_type="ligand", label="cand")
    gen = DrugFlowGenerator(_config())
    gen.assign(ligand)
    ligand.proposal_sequences = [
        type(ligand.proposal_sequences[0])(sequence="", sequence_type="ligand") for _ in range(3)
    ]
    with patch(f"{MODULE}.run_drugflow_generate", return_value=_fake_output(["CCO", "CCN"])):
        gen.sample()
    assert [p.sequence for p in ligand.proposal_sequences] == ["CCO", "CCN", "CCO"]


def test_sample_raises_on_empty_output():
    """An empty tool result is a hard error rather than blank proposals."""
    ligand = Segment(sequence="", sequence_type="ligand", label="cand")
    gen = DrugFlowGenerator(_config())
    gen.assign(ligand)
    with patch(f"{MODULE}.run_drugflow_generate", return_value=DrugFlowGenerateOutput(generation_sets=[])):
        with pytest.raises(RuntimeError, match="no molecules"):
            gen.sample()


def test_config_rejects_invalid_pocket_cutoff():
    """A non-positive pocket distance cutoff is rejected."""
    with pytest.raises(ValueError):
        _config(pocket_distance_cutoff=0.0)
