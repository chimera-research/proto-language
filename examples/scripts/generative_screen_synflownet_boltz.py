"""Generative small-molecule screen: SynFlowNet ligand generation + Boltz-2 affinity scoring.

This program runs a generative screen against a fixed protein target in the Proto abstractions:

- A **ligand** ``Segment`` is designed de novo by the ``SynFlowNetGenerator``, which samples
  synthesizable molecules (each with a synthesis route) from a trained SynFlowNet GFlowNet.
- A **protein** ``Segment`` holds the fixed target and carries no generator.
- A ``boltz2_affinity_constraint`` wires the two segments into a protein-ligand complex and scores
  each sampled molecule by Boltz-2's predicted binding affinity (log10 IC50) and binder probability.
- A ``RejectionSamplingOptimizer`` samples a batch of molecules, scores them, and keeps the
  top-scoring hits — i.e. a generative screen.

Protein and ligand live in separate single-segment ``Construct``s (a construct is single-type);
the multi-input affinity constraint is where the protein and ligand meet.

This mirrors the SynFlowNet + Boltz-2 generative-screen setup, expressed with Proto's Sequence /
Generator / Constraint / Optimizer / Program abstractions.

Running it requires a GPU, the Boltz-2 weights (auto-provisioned), and a trained SynFlowNet
checkpoint (user-provisioned; see the proto-tools synflownet toolkit README). Use ``--build-only``
to construct and validate the program without running it.

Example:
    python examples/scripts/generative_screen_synflownet_boltz.py --num-samples 64 --num-hits 8
    python examples/scripts/generative_screen_synflownet_boltz.py --build-only
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

# Example target: the B1 domain of streptococcal protein G (GB1), a small, well-characterized
# protein used here as a self-contained stand-in. Replace it with your real target sequence
# (e.g. a kinase, protease, or GPCR domain) via --target-sequence.
DEFAULT_TARGET_SEQUENCE = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


def build_program(args: argparse.Namespace) -> tuple[Program, Segment]:
    """Build the single-stage generative screen program.

    Args:
        args (argparse.Namespace): Parsed command-line options.

    Returns:
        tuple[Program, Segment]: The program and the ligand segment to read hits from.
    """
    # Step 1 — Segments: a fixed protein target and an empty ligand to design de novo.
    target = Segment(sequence=args.target_sequence, sequence_type="protein", label="target")
    ligand = Segment(sequence="", sequence_type="ligand", label="candidate")

    # Step 2 — Constructs: protein and ligand are each a single-type construct.
    target_construct = Construct([target])
    ligand_construct = Construct([ligand])

    # Step 3 — Generator: SynFlowNet proposes synthesizable molecules into the ligand segment.
    generator = SynFlowNetGenerator(
        SynFlowNetGeneratorConfig(
            model_checkpoint=args.checkpoint,
            objective=args.objective,
            sampling_temperature=args.sampling_temperature,
            max_reaction_steps=args.max_reaction_steps,
            batch_size=args.num_samples,
            device=args.device,
            verbose=args.verbose,
        )
    )
    generator.assign(ligand)

    # Step 4 — Constraint: score each molecule by Boltz-2 predicted affinity to the target.
    affinity = Constraint(
        inputs=[target, ligand],
        function=boltz2_affinity_constraint,
        function_config=Boltz2AffinityConstraintConfig(
            return_component="binding_penalty",
            desired_affinity_value=args.desired_affinity_value,
            boltz2_affinity_config={"device": args.device, "verbose": int(args.verbose)},
        ),
        label="Boltz2 predicted affinity",
    )

    # Step 5 — Optimizer: sample a batch, score, keep the top hits (the screen).
    screen = RejectionSamplingOptimizer(
        constructs=[target_construct, ligand_construct],
        generators=[generator],
        constraints=[affinity],
        config=RejectionSamplingOptimizerConfig(
            num_samples=args.num_samples,
            num_results=args.num_hits,
            proposal_batch_size=args.num_samples,
            verbose=args.verbose,
            seed=args.seed,
        ),
    )

    # Step 6 — Program.
    program = Program(optimizers=[screen], num_results=args.num_hits, seed=args.seed)
    return program, ligand


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-sequence", default=DEFAULT_TARGET_SEQUENCE, help="Target protein sequence.")
    parser.add_argument("--checkpoint", default="default", help="SynFlowNet checkpoint ('default' or a .pt path).")
    parser.add_argument("--objective", default=None, help="Optional named reward objective for the checkpoint.")
    parser.add_argument("--num-samples", type=int, default=64, help="Molecules to sample and score per screen.")
    parser.add_argument("--num-hits", type=int, default=8, help="Top-scoring molecules to retain.")
    parser.add_argument("--sampling-temperature", type=float, default=32.0, help="GFlowNet inverse-temperature.")
    parser.add_argument("--max-reaction-steps", type=int, default=4, help="Maximum synthesis steps per molecule.")
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
    """Build and run the generative screen, then print the retained hits."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    program, ligand = build_program(args)
    logger.info("Built generative screen: %d samples -> top %d hits", args.num_samples, args.num_hits)

    if args.build_only:
        logger.info("Build-only: program constructed and validated; skipping the run.")
        return

    program.run()

    logger.info("Top %d ligands (SMILES, predicted-affinity penalty):", ligand.num_results)
    for rank, sequence in enumerate(ligand.result_sequences, start=1):
        affinity_meta = sequence.metadata.get("constraints", {}).get("Boltz2 predicted affinity", {})
        generator_meta = sequence.metadata.get("generators", {}).get("synflownet", {})
        logger.info(
            "  %2d. %s | affinity=%s | reward=%s",
            rank,
            sequence.sequence,
            affinity_meta,
            generator_meta.get("reward"),
        )


if __name__ == "__main__":
    main()
