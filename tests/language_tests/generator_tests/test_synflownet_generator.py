"""Tests for SynFlowNetGenerator (de novo ligand generation)."""

from unittest.mock import patch

import pytest
from proto_tools import (
    SynFlowNetMolecule,
    SynFlowNetMoleculeMetrics,
    SynFlowNetSampleOutput,
    SynFlowNetSampleSet,
)

from proto_language.core import Segment
from proto_language.generator import SynFlowNetGenerator, SynFlowNetGeneratorConfig
from proto_language.generator.generator_registry import GeneratorRegistry

MODULE = "proto_language.generator.synflownet_generator"


def _fake_output(smiles: list[str]) -> SynFlowNetSampleOutput:
    """Build a real SynFlowNetSampleOutput with one sample set of the given molecules."""
    molecules = [
        SynFlowNetMolecule(
            smiles=smi,
            synthesis_route=["rxn-1"],
            building_blocks=["CCO"],
            metrics=SynFlowNetMoleculeMetrics(reward=1.0 + i, qed=0.5, molecular_weight=120.0, num_reactions=1.0),
        )
        for i, smi in enumerate(smiles)
    ]
    return SynFlowNetSampleOutput(sample_sets=[SynFlowNetSampleSet(molecules=molecules)])


def test_init_stores_config_values():
    """Generator stores config values and declares the ligand/de-novo contract."""
    config = SynFlowNetGeneratorConfig(sampling_temperature=16.0, max_reaction_steps=6, batch_size=8, objective="qed")
    gen = SynFlowNetGenerator(config)
    assert gen.sampling_temperature == 16.0
    assert gen.max_reaction_steps == 6
    assert gen.batch_size == 8
    assert gen.objective == "qed"
    assert gen.generates_ligands is True
    assert gen.allows_empty_starting_sequence is True


def test_registered_as_ligand_generator():
    """The generator is registered with the ligand sequence type and mutation category."""
    spec = GeneratorRegistry.get("synflownet")
    assert spec.supported_sequence_types == ["ligand"]
    assert spec.category == "mutation"
    assert spec.allows_empty_starting_sequence is True


def test_assign_accepts_empty_ligand_segment():
    """A de novo ligand segment (empty SMILES) accepts the generator."""
    ligand = Segment(sequence="", sequence_type="ligand", label="cand")
    gen = SynFlowNetGenerator(SynFlowNetGeneratorConfig())
    gen.assign(ligand)
    assert gen.is_assigned
    assert gen.segment is ligand


@pytest.mark.parametrize("sequence_type", ["protein", "dna", "rna"])
def test_rejects_non_ligand_segment(sequence_type):
    """The generator only supports ligand segments."""
    seq = {"protein": "ACDEFG", "dna": "ACGT", "rna": "ACGU"}[sequence_type]
    segment = Segment(sequence=seq, sequence_type=sequence_type)
    gen = SynFlowNetGenerator(SynFlowNetGeneratorConfig())
    with pytest.raises(ValueError, match="does not support sequence type"):
        gen.assign(segment)


def test_sample_writes_smiles_and_metadata():
    """_sample fills the proposal pool with sampled SMILES and records per-proposal metadata."""
    ligand = Segment(sequence="", sequence_type="ligand", label="cand")
    gen = SynFlowNetGenerator(SynFlowNetGeneratorConfig())
    gen.assign(ligand)
    # Size the proposal pool to 3 (what an optimizer would do before sampling).
    ligand.proposal_sequences = [type(ligand.proposal_sequences[0])(sequence="", sequence_type="ligand") for _ in range(3)]

    with patch(f"{MODULE}.run_synflownet_sample", return_value=_fake_output(["CCO", "c1ccccc1", "CCN"])) as mock_run:
        gen.sample()

    assert [p.sequence for p in ligand.proposal_sequences] == ["CCO", "c1ccccc1", "CCN"]
    meta = ligand.proposal_sequences[0]._generator_metadata["synflownet"]
    assert meta["reward"] == 1.0
    assert meta["synthesis_route"] == ["rxn-1"]
    assert meta["building_blocks"] == ["CCO"]
    # The tool was asked for exactly num_proposals molecules.
    assert mock_run.call_args.kwargs["config"].num_molecules_per_request == 3


def test_sample_cycles_when_tool_returns_fewer():
    """When the tool returns fewer molecules than proposals, they are cycled to fill the pool."""
    ligand = Segment(sequence="", sequence_type="ligand", label="cand")
    gen = SynFlowNetGenerator(SynFlowNetGeneratorConfig())
    gen.assign(ligand)
    ligand.proposal_sequences = [type(ligand.proposal_sequences[0])(sequence="", sequence_type="ligand") for _ in range(3)]

    with patch(f"{MODULE}.run_synflownet_sample", return_value=_fake_output(["CCO", "CCN"])):
        gen.sample()

    assert [p.sequence for p in ligand.proposal_sequences] == ["CCO", "CCN", "CCO"]


def test_sample_raises_on_empty_tool_output():
    """An empty tool result is a hard error rather than leaving blank proposals."""
    ligand = Segment(sequence="", sequence_type="ligand", label="cand")
    gen = SynFlowNetGenerator(SynFlowNetGeneratorConfig())
    gen.assign(ligand)

    with patch(f"{MODULE}.run_synflownet_sample", return_value=SynFlowNetSampleOutput(sample_sets=[])):
        with pytest.raises(RuntimeError, match="no molecules"):
            gen.sample()


def test_config_rejects_invalid_values():
    """Config bounds reject a non-positive temperature and out-of-range action probability."""
    with pytest.raises(ValueError):
        SynFlowNetGeneratorConfig(sampling_temperature=0.0)
    with pytest.raises(ValueError):
        SynFlowNetGeneratorConfig(random_action_prob=1.5)
