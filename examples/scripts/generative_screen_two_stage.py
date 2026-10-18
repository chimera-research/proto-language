"""Two-stage generative screen: broad SynFlowNet sampling, then high-fidelity Boltz-2 re-scoring.

A funnel-style generative screen that trades compute for throughput the way real screens do:

- **Stage 1 (broad):** the ``SynFlowNetGenerator`` samples a large batch of synthesizable
  molecules, scored with a fast, low-fidelity Boltz-2 affinity pass (few diffusion samples). The
  optimizer keeps the top ``--stage1-keep`` hits.
- **Stage 2 (refine):** those hits are re-scored with a high-fidelity Boltz-2 affinity pass (more
  diffusion samples / steps) using ``proposal_source="existing_results"`` (no new generation), and
  the top ``--num-hits`` are retained.

Both optimizer stages share the **same** construct objects by identity, so the Stage 1 hits flow
into Stage 2 through the ligand segment's result pool — the canonical Proto multi-stage pattern.

Requires a GPU, Boltz-2 weights, and a trained SynFlowNet checkpoint. Use ``--build-only`` to
construct and validate without running.

Example:
    python examples/scripts/generative_screen_two_stage.py --num-samples 128 --stage1-keep 16 --num-hits 4
    python examples/scripts/generative_screen_two_stage.py --build-only
"""

import argparse
import logging

from proto_language.constraint.protein_structure.boltz2_affinity_constraint import (
    Boltz2AffinityConstraintConfig,
    boltz2_affinity_constraint,
)
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import SynFlowNetGenerator, SynFlowNetGeneratorConfig
from proto_language.optimizer import RejectionSamplingOptimizer, RejectionSamplingOptimizerConfig

logger = logging.getLogger(__name__)

# Example stand-in target (GB1); replace with your real target via --target-sequence.
DEFAULT_TARGET_SEQUENCE = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


def _affinity_constraint(
    target: Segment, ligand: Segment, args: argparse.Namespace, diffusion_samples: int, label: str
) -> Constraint:
    """Build a Boltz-2 affinity constraint at a given fidelity (affinity diffusion samples)."""
    return Constraint(
        inputs=[target, ligand],
        function=boltz2_affinity_constraint,
        function_config=Boltz2AffinityConstraintConfig(
            return_component="binding_penalty",
            desired_affinity_value=args.desired_affinity_value,
            boltz2_affinity_config={
                "device": args.device,
                "verbose": int(args.verbose),
                "diffusion_samples_affinity": diffusion_samples,
            },
        ),
        label=label,
    )


def build_program(args: argparse.Namespace) -> tuple[Program, Segment]:
    """Build the two-stage broad-then-refine generative screen program.

    Args:
        args (argparse.Namespace): Parsed command-line options.

    Returns:
        tuple[Program, Segment]: The program and the ligand segment to read hits from.
    """
    # Shared segments and constructs — reused by identity across both stages.
    target = Segment(sequence=args.target_sequence, sequence_type="protein", label="target")
    ligand = Segment(sequence="", sequence_type="ligand", label="candidate")
    constructs = [Construct([target]), Construct([ligand])]

    # Stage 1 — broad sampling with a fast, low-fidelity affinity pass.
    stage1_generator = SynFlowNetGenerator(
        SynFlowNetGeneratorConfig(
            model_checkpoint=args.checkpoint,
            sampling_temperature=args.sampling_temperature,
            batch_size=args.num_samples,
            device=args.device,
            verbose=args.verbose,
        )
    )
    stage1_generator.assign(ligand)
    stage1 = RejectionSamplingOptimizer(
        constructs=constructs,
        generators=[stage1_generator],
        constraints=[_affinity_constraint(target, ligand, args, diffusion_samples=1, label="Boltz2 affinity (fast)")],
        config=RejectionSamplingOptimizerConfig(
            num_samples=args.num_samples,
            num_results=args.stage1_keep,
            proposal_batch_size=args.num_samples,
            verbose=args.verbose,
            seed=args.seed,
        ),
    )

    # Stage 2 — re-score the Stage 1 hits with a high-fidelity affinity pass (no new generation).
    stage2 = RejectionSamplingOptimizer(
        constructs=constructs,
        generators=[],
        constraints=[
            _affinity_constraint(
                target, ligand, args, diffusion_samples=args.refine_diffusion_samples, label="Boltz2 affinity (refined)"
            )
        ],
        config=RejectionSamplingOptimizerConfig(
            num_samples=args.stage1_keep,
            num_results=args.num_hits,
            proposal_source="existing_results",
            verbose=args.verbose,
            seed=args.seed,
        ),
    )

    program = Program(optimizers=[stage1, stage2], num_results=args.num_hits, seed=args.seed)
    return program, ligand


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-sequence", default=DEFAULT_TARGET_SEQUENCE, help="Target protein sequence.")
    parser.add_argument("--checkpoint", default="default", help="SynFlowNet checkpoint ('default' or a .pt path).")
    parser.add_argument("--num-samples", type=int, default=128, help="Molecules sampled in the broad Stage 1.")
    parser.add_argument("--stage1-keep", type=int, default=16, help="Hits carried from Stage 1 into Stage 2.")
    parser.add_argument("--num-hits", type=int, default=4, help="Final refined hits to retain.")
    parser.add_argument("--sampling-temperature", type=float, default=32.0, help="GFlowNet inverse-temperature.")
    parser.add_argument(
        "--refine-diffusion-samples",
        type=int,
        default=5,
        help="Boltz-2 affinity diffusion samples in the high-fidelity Stage 2.",
    )
    parser.add_argument(
        "--desired-affinity-value",
        type=float,
        default=-2.0,
        help="Target predicted affinity (log10 IC50 uM); lower is stronger.",
    )
    parser.add_argument("--device", default="cuda", help="Device for Boltz-2 and SynFlowNet.")
    parser.add_argument("--seed", type=int, default=0, help="Program seed for reproducibility.")
    parser.add_argument("--verbose", action="store_true", help="Verbose tool output.")
    parser.add_argument("--build-only", action="store_true", help="Build and validate the program without running it.")
    return parser.parse_args()


def main() -> None:
    """Build and run the two-stage screen, then print the final refined hits."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    program, ligand = build_program(args)
    logger.info(
        "Built two-stage screen: %d sampled -> %d after Stage 1 -> top %d refined",
        args.num_samples,
        args.stage1_keep,
        args.num_hits,
    )

    if args.build_only:
        logger.info("Build-only: program constructed and validated; skipping the run.")
        return

    program.run()

    logger.info("Top %d refined ligands (SMILES):", ligand.num_results)
    for rank, sequence in enumerate(ligand.result_sequences, start=1):
        logger.info("  %2d. %s", rank, sequence.sequence)


if __name__ == "__main__":
    main()
