"""SynFlowNetGenerator for de novo generation of synthesizable small-molecule ligands.

A ``SynFlowNetGenerator`` proposes novel ligand SMILES into a ligand ``Segment`` by sampling a
trained `SynFlowNet <https://github.com/mirunacrt/synflownet>`_ GFlowNet, which builds molecules
from chemical reactions and purchasable building blocks (so every molecule comes with a synthesis
route). It is the generator half of a generative screen: pair it with a scoring constraint such as
``boltz2-affinity`` to rank the sampled ligands against a protein target. Unlike sequence
generators, it declares ``generates_ligands = True`` and so may be assigned to a ligand segment,
and ``allows_empty_starting_sequence = True`` so it can design from an empty ligand segment.

Examples:
    De novo sample ligands into an empty ligand segment:

    >>> from proto_language.core import Segment
    >>> from proto_language.generator import SynFlowNetGenerator, SynFlowNetGeneratorConfig
    >>> ligand = Segment(sequence="", sequence_type="ligand", label="candidate")
    >>> gen = SynFlowNetGenerator(SynFlowNetGeneratorConfig())
    >>> gen.assign(ligand)
    >>> gen.generates_ligands  # True
"""

import logging
from typing import final

from proto_tools import (
    SynFlowNetRequest,
    SynFlowNetSampleConfig,
    SynFlowNetSampleInput,
    run_synflownet_sample,
)

from proto_language.core import Generator, GeneratorInputType
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField

logger = logging.getLogger(__name__)


class SynFlowNetGeneratorConfig(BaseConfig):
    """Configuration object for SynFlowNetGenerator.

    Attributes:
        model_checkpoint (str): Trained SynFlowNet GFlowNet checkpoint. ``"default"`` uses the
            checkpoint provisioned in the managed weights cache; any other value is a path to a
            ``.pt`` checkpoint.
        objective (str | None): Optional named reward objective for objective-conditioned
            checkpoints (e.g. ``"seh"``, ``"qed"``); ``None`` samples from the checkpoint's
            native reward.
        sampling_temperature (float): GFlowNet inverse-temperature (beta); higher concentrates
            sampling on high-reward molecules, lower broadens chemical diversity.
        random_action_prob (float): Probability of a uniform-random synthesis action per step,
            for extra exploration.
        max_reaction_steps (int): Maximum number of synthesis actions per molecule.
        unique_molecules (bool): Deduplicate identical canonical SMILES within a sampling call.
            Defaults to ``False`` so exactly ``num_proposals`` molecules are returned per call.
        tool_seed (int | None): Optional seed passed directly to the SynFlowNet tool; ``None``
            uses Proto's derived seed stream.
        batch_size (int): Molecules sampled per GFlowNet forward pass; raise for throughput,
            lower if out of GPU memory.
        device (str): GPU device for inference (e.g. ``"cuda"`` or ``"cuda:0"``).
        verbose (bool): Whether to print status messages during execution.
    """

    model_checkpoint: str = ConfigField(
        default="default",
        title="Model Checkpoint",
        description="'default' uses the provisioned checkpoint, or a path to a .pt SynFlowNet checkpoint.",
    )
    objective: str | None = ConfigField(
        default=None,
        title="Objective",
        description="Named reward objective for multi-objective checkpoints; None uses the native reward.",
    )
    sampling_temperature: float = ConfigField(
        default=32.0,
        gt=0.0,
        title="Sampling Temperature",
        description="GFlowNet inverse-temperature (beta); higher favors high-reward molecules.",
    )
    random_action_prob: float = ConfigField(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Random Action Probability",
        description="Probability of a uniform-random synthesis action per step, for exploration.",
    )
    max_reaction_steps: int = ConfigField(
        default=4,
        ge=1,
        title="Max Reaction Steps",
        description="Maximum synthesis actions per molecule (trajectory length).",
    )
    unique_molecules: bool = ConfigField(
        default=False,
        title="Unique Molecules",
        description="Deduplicate identical canonical SMILES within a sampling call.",
    )
    tool_seed: int | None = ConfigField(
        default=None,
        ge=0,
        title="Tool Seed",
        description="Optional seed passed directly to the SynFlowNet tool; None uses Proto's derived seed stream.",
    )
    batch_size: int = ConfigField(
        default=64,
        ge=1,
        title="Batch Size",
        description="Molecules sampled per GFlowNet forward pass; raise for throughput, lower if OOM.",
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


@generator(
    key="synflownet",
    label="SynFlowNet Molecule Generation",
    config=SynFlowNetGeneratorConfig,
    description="De novo generation of synthesizable small-molecule ligands with a SynFlowNet GFlowNet",
    uses_gpu=True,
    tools_called=["synflownet-sample"],
    supported_sequence_types=["ligand"],
)
@final
class SynFlowNetGenerator(Generator):
    """Ligand generator that samples synthesizable small molecules from a SynFlowNet GFlowNet.

    SynFlowNet builds molecules from chemical reactions and purchasable building blocks, sampling
    in proportion to a reward, so generated molecules are synthesizable by construction and carry a
    synthesis route. This generator writes sampled SMILES into a ligand segment's proposal pool and
    records each molecule's reward, synthesis route, building blocks, and RDKit properties in the
    per-proposal generator metadata. It is designed to drive generative screens: an optimizer
    selects among the proposed ligands using scoring constraints (e.g. ``boltz2-affinity`` against
    a fixed protein target).

    The generator category is ``"mutation"`` (de novo). Each ``sample()`` call requests
    ``num_proposals`` molecules from the tool and fills the ligand segment's proposal pool with
    them; it does not mutate a starting molecule, so the segment may be created empty
    (``Segment(sequence="", sequence_type="ligand")``).

    Attributes:
        batch_size (int): Number of molecules sampled per GFlowNet forward pass.

    Example:
        >>> from proto_language.generator import SynFlowNetGenerator, SynFlowNetGeneratorConfig
        >>> from proto_language.core import Segment
        >>> config = SynFlowNetGeneratorConfig(sampling_temperature=32.0, max_reaction_steps=4)
        >>> gen = SynFlowNetGenerator(config)
        >>> ligand = Segment(sequence="", sequence_type="ligand", label="candidate")
        >>> gen.assign(ligand)
        >>> gen.sample()  # Fills the proposal pool with sampled synthesizable molecules
    """

    input_type = GeneratorInputType.STARTING_SEQUENCE
    allows_empty_starting_sequence = True
    generates_ligands = True

    def __init__(self, config: SynFlowNetGeneratorConfig) -> None:
        """Initialize the SynFlowNet generator.

        Args:
            config (SynFlowNetGeneratorConfig): Configuration object containing all generator parameters.
        """
        super().__init__()
        self.config = config
        self.model_checkpoint = config.model_checkpoint
        self.objective = config.objective
        self.sampling_temperature = config.sampling_temperature
        self.random_action_prob = config.random_action_prob
        self.max_reaction_steps = config.max_reaction_steps
        self.unique_molecules = config.unique_molecules
        self.tool_seed = config.tool_seed
        self.batch_size = config.batch_size
        self.device = config.device
        self.verbose = config.verbose

    def _sample(self) -> None:
        """Sample synthesizable molecules and write them into the ligand proposal pool.

        Requests ``num_proposals`` molecules from the SynFlowNet tool as a single request, then
        overwrites each proposal's SMILES with a sampled molecule and records its reward, synthesis
        route, building blocks, and RDKit properties under
        ``proposal._generator_metadata[self._spec.key]``. When the tool returns fewer molecules
        than proposals (e.g. after uniqueness filtering), the returned molecules are cycled to fill
        the pool.

        Raises:
            RuntimeError: If called before assign(), or if the tool returns no molecules.
        """
        self._validate_generator()
        num_proposals = self.segment.num_proposals

        tool_config = SynFlowNetSampleConfig(
            model_checkpoint=self.model_checkpoint,
            num_molecules_per_request=num_proposals,
            sampling_temperature=self.sampling_temperature,
            random_action_prob=self.random_action_prob,
            max_reaction_steps=self.max_reaction_steps,
            unique_molecules=self.unique_molecules,
            batch_size=self.batch_size,
            seed=self.tool_seed if self.tool_seed is not None else self._next_seed(),
            device=self.device,
            verbose=self.verbose,
        )
        result = run_synflownet_sample(
            inputs=SynFlowNetSampleInput(requests=[SynFlowNetRequest(objective=self.objective)]),
            config=tool_config,
        )

        molecules = result.sample_sets[0].molecules if result.sample_sets else []
        if not molecules:
            raise RuntimeError(
                f"{self.__class__.__name__}: SynFlowNet returned no molecules for segment "
                f"{(self.segment.label or 'unlabeled')!r}."
            )
        if len(molecules) < num_proposals:
            logger.warning(
                "%s: SynFlowNet returned %d molecules for %d proposals; cycling to fill the pool.",
                self.__class__.__name__,
                len(molecules),
                num_proposals,
            )

        key = self._spec.key
        for index, proposal in enumerate(self.segment.proposal_sequences):
            molecule = molecules[index % len(molecules)]
            proposal.sequence = molecule.smiles
            proposal._generator_metadata[key] = {
                "reward": molecule.metrics.get("reward"),
                "qed": molecule.metrics.get("qed"),
                "molecular_weight": molecule.metrics.get("molecular_weight"),
                "num_reactions": molecule.metrics.get("num_reactions"),
                "synthesis_route": molecule.synthesis_route,
                "building_blocks": molecule.building_blocks,
                "objective": self.objective,
            }
