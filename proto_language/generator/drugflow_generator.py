"""DrugFlowGenerator for pocket-conditioned de novo generation of small-molecule ligands.

A ``DrugFlowGenerator`` proposes novel ligand SMILES into a ligand ``Segment`` by sampling
`DrugFlow <https://github.com/LPDI-EPFL/DrugFlow>`_, a structure-based generative model that
combines continuous flow matching with discrete Markov bridges to generate 3D molecules inside a
protein pocket. Unlike the (reward-trained) SynFlowNet generator, DrugFlow ships public checkpoints
and is conditioned directly on a target: given a protein structure and a reference ligand that marks
the pocket, it generates posed molecules ready for inference. It is the generator half of an
inference-ready generative screen — pair it with a scoring constraint such as ``boltz2-affinity`` or
``boltz2-binding-strength`` to rank the generated ligands against the target.

Like other de novo ligand generators it declares ``generates_ligands = True`` (so it may be assigned
to a ligand segment) and ``allows_empty_starting_sequence = True`` (so it can design from an empty
ligand segment). Each generated molecule's 3D pose and RDKit properties are recorded in the
per-proposal generator metadata.

Examples:
    De novo generate ligands into an empty ligand segment, conditioned on a target pocket:

    >>> from proto_language.core import Segment
    >>> from proto_language.generator import DrugFlowGenerator, DrugFlowGeneratorConfig
    >>> ligand = Segment(sequence="", sequence_type="ligand", label="candidate")
    >>> config = DrugFlowGeneratorConfig(target_structure="target.pdb", reference_ligand="ref.sdf")
    >>> gen = DrugFlowGenerator(config)
    >>> gen.assign(ligand)
    >>> gen.generates_ligands  # True
"""

import logging
from typing import Any, final

from proto_tools import (
    DrugFlowGenerateConfig,
    DrugFlowGenerateInput,
    run_drugflow_generate,
)
from proto_tools.entities.structures import Structure, load_structure_file
from pydantic import field_validator

from proto_language.core import Generator, GeneratorInputType
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField

logger = logging.getLogger(__name__)


class DrugFlowGeneratorConfig(BaseConfig):
    """Configuration object for DrugFlowGenerator.

    Attributes:
        target_structure (Structure): Target protein structure whose pocket conditions generation.
            Accepts a ``Structure``, a path to a PDB/CIF file, or raw PDB content.
        reference_ligand (str): Reference ligand defining the pocket, as an SDF block or a path to an
            ``.sdf`` file with 3D coordinates; the pocket is the residues within
            ``pocket_distance_cutoff`` of it.
        model_checkpoint (str): Bundled DrugFlow checkpoint name (``drugflow``, ``drugflow_ood``,
            ``flexflow``, or ``drugflow_pa_comb``), or a path to a custom ``.ckpt``.
        pocket_distance_cutoff (float): Distance (Å) around the reference ligand that defines the pocket.
        molecule_size (int | None): Maximum heavy-atom count per molecule; ``None`` samples sizes from
            DrugFlow's training-set distribution.
        n_steps (int | None): Number of denoising steps; ``None`` uses the checkpoint default.
        tool_seed (int | None): Optional seed passed directly to the DrugFlow tool; ``None`` uses
            Proto's derived seed stream.
        batch_size (int): Molecules sampled per GPU forward pass; raise for throughput, lower if OOM.
        device (str): GPU device for inference (e.g. ``"cuda"`` or ``"cuda:0"``).
        verbose (bool): Whether to print status messages during execution.
    """

    target_structure: Structure = ConfigField(
        title="Target Structure",
        description="Target protein structure (Structure, PDB/CIF path, or PDB content) defining the pocket.",
    )
    reference_ligand: str = ConfigField(
        title="Reference Ligand",
        description="Reference ligand (SDF block or .sdf path) whose neighborhood defines the pocket.",
    )
    model_checkpoint: str = ConfigField(
        default="drugflow",
        title="Model Checkpoint",
        description="Bundled name ('drugflow', 'drugflow_ood', 'flexflow', 'drugflow_pa_comb') or a .ckpt path.",
    )
    pocket_distance_cutoff: float = ConfigField(
        default=8.0,
        gt=0.0,
        title="Pocket Distance Cutoff",
        description="Distance (Å) around the reference ligand that defines the pocket.",
    )
    molecule_size: int | None = ConfigField(
        default=None,
        ge=1,
        title="Molecule Size",
        description="Max heavy atoms per molecule; None samples from DrugFlow's size distribution.",
    )
    n_steps: int | None = ConfigField(
        default=None,
        ge=1,
        title="Denoising Steps",
        description="Number of denoising steps; None uses the checkpoint default.",
    )
    tool_seed: int | None = ConfigField(
        default=None,
        ge=0,
        title="Tool Seed",
        description="Optional seed passed directly to the DrugFlow tool; None uses Proto's derived seed stream.",
    )
    batch_size: int = ConfigField(
        default=32,
        ge=1,
        title="Batch Size",
        description="Molecules sampled per GPU forward pass; raise for throughput, lower if OOM.",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="GPU device for inference (e.g. 'cuda' or 'cuda:0').",
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print status messages during execution.",
    )

    @field_validator("target_structure", mode="before")
    @classmethod
    def normalize_target_structure(cls, value: Any) -> Any:
        """Convert a path or raw PDB content to a ``Structure``."""
        if isinstance(value, (Structure, dict)):
            return value
        if isinstance(value, str):
            if "\n" not in value and value.strip().lower().endswith((".pdb", ".cif")):
                return load_structure_file(value)
            return Structure(structure=value, structure_format="pdb")
        return value


@generator(
    key="drugflow",
    label="DrugFlow Molecule Generation",
    config=DrugFlowGeneratorConfig,
    description="Pocket-conditioned de novo generation of 3D small-molecule ligands with DrugFlow",
    uses_gpu=True,
    tools_called=["drugflow-generate"],
    supported_sequence_types=["ligand"],
)
@final
class DrugFlowGenerator(Generator):
    """Ligand generator that samples pocket-conditioned 3D molecules from DrugFlow.

    DrugFlow generates molecules inside a protein pocket (defined by a target structure plus a
    reference ligand), so its proposals are target-aware by construction. This generator writes the
    generated SMILES into a ligand segment's proposal pool and records each molecule's 3D pose and
    RDKit properties in the per-proposal generator metadata. It drives inference-ready generative
    screens: an optimizer selects among the proposed ligands using scoring constraints (e.g.
    ``boltz2-affinity`` against the same target).

    The generator category is ``"mutation"`` (de novo). Each ``sample()`` call requests
    ``num_proposals`` molecules from the tool and fills the ligand segment's proposal pool with them;
    it does not mutate a starting molecule, so the segment may be created empty
    (``Segment(sequence="", sequence_type="ligand")``).

    Attributes:
        batch_size (int): Number of molecules sampled per GPU forward pass.

    Example:
        >>> from proto_language.generator import DrugFlowGenerator, DrugFlowGeneratorConfig
        >>> from proto_language.core import Segment
        >>> config = DrugFlowGeneratorConfig(target_structure="target.pdb", reference_ligand="ref.sdf")
        >>> gen = DrugFlowGenerator(config)
        >>> ligand = Segment(sequence="", sequence_type="ligand", label="candidate")
        >>> gen.assign(ligand)
        >>> gen.sample()  # Fills the proposal pool with pocket-conditioned molecules
    """

    input_type = GeneratorInputType.STARTING_SEQUENCE
    allows_empty_starting_sequence = True
    generates_ligands = True

    def __init__(self, config: DrugFlowGeneratorConfig) -> None:
        """Initialize the DrugFlow generator.

        Args:
            config (DrugFlowGeneratorConfig): Configuration object containing all generator parameters.
        """
        super().__init__()
        self.config = config
        self.target_structure = config.target_structure
        self.reference_ligand = config.reference_ligand
        self.model_checkpoint = config.model_checkpoint
        self.pocket_distance_cutoff = config.pocket_distance_cutoff
        self.molecule_size = config.molecule_size
        self.n_steps = config.n_steps
        self.tool_seed = config.tool_seed
        self.batch_size = config.batch_size
        self.device = config.device
        self.verbose = config.verbose

    def _sample(self) -> None:
        """Generate pocket-conditioned molecules and write them into the ligand proposal pool.

        Requests ``num_proposals`` molecules from the DrugFlow tool for the configured target pocket,
        then overwrites each proposal's SMILES with a generated molecule and records its 3D pose and
        RDKit properties under ``proposal._generator_metadata[self._spec.key]``. When the tool
        returns fewer molecules than proposals, the returned molecules are cycled to fill the pool.

        Raises:
            RuntimeError: If called before assign(), or if the tool returns no molecules.
        """
        self._validate_generator()
        num_proposals = self.segment.num_proposals

        tool_config = DrugFlowGenerateConfig(
            model_checkpoint=self.model_checkpoint,
            num_molecules_per_target=num_proposals,
            batch_size=self.batch_size,
            pocket_distance_cutoff=self.pocket_distance_cutoff,
            molecule_size=self.molecule_size,
            n_steps=self.n_steps,
            seed=self.tool_seed if self.tool_seed is not None else self._next_seed(),
            device=self.device,
            verbose=self.verbose,
        )
        result = run_drugflow_generate(
            inputs=DrugFlowGenerateInput(
                proteins=[self.target_structure],
                reference_ligands=[self.reference_ligand],
            ),
            config=tool_config,
        )

        molecules = result.generation_sets[0].molecules if result.generation_sets else []
        if not molecules:
            raise RuntimeError(
                f"{self.__class__.__name__}: DrugFlow returned no molecules for segment "
                f"{(self.segment.label or 'unlabeled')!r}."
            )
        if len(molecules) < num_proposals:
            logger.warning(
                "%s: DrugFlow returned %d molecules for %d proposals; cycling to fill the pool.",
                self.__class__.__name__,
                len(molecules),
                num_proposals,
            )

        key = self._spec.key
        for index, proposal in enumerate(self.segment.proposal_sequences):
            molecule = molecules[index % len(molecules)]
            proposal.sequence = molecule.smiles
            proposal._generator_metadata[key] = {
                "qed": molecule.metrics.get("qed"),
                "molecular_weight": molecule.metrics.get("molecular_weight"),
                "num_atoms": molecule.metrics.get("num_atoms"),
                "pose_sdf": molecule.pose_sdf,
                "model_checkpoint": self.model_checkpoint,
            }
