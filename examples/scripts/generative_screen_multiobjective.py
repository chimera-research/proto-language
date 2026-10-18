"""Multi-objective generative screen: SynFlowNet + Boltz-2 affinity AND complex confidence.

A richer generative screen than the single-objective version: each de novo molecule sampled by
SynFlowNet is scored against the protein target by **two** Boltz-2 functions, composed as weighted
scoring constraints on the same protein-ligand complex:

- ``boltz2_affinity_constraint`` — predicted binding affinity (log10 IC50) and binder probability.
- ``boltz_binding_strength_constraint`` (``return_component="ligand_iptm"``) — Boltz-2's predicted
  ligand-interface confidence, a structural check that the complex is modelled confidently.

The optimizer's energy is the weighted sum of the two penalties, so retained hits are molecules
that are both predicted to bind tightly and to form a confidently-modelled complex. This shows how
Proto composes multiple scoring functions over the same design, using both of Boltz-2's
functionalities (affinity prediction and complex structure prediction).

Requires a GPU, Boltz-2 weights, and a trained SynFlowNet checkpoint. Use ``--build-only`` to
construct and validate without running.

Example:
    python examples/scripts/generative_screen_multiobjective.py --num-samples 64 --num-hits 8
    python examples/scripts/generative_screen_multiobjective.py --build-only
"""

import argparse
import logging

from proto_language.constraint.protein_structure.boltz2_affinity_constraint import (
    Boltz2AffinityConstraintConfig,
    boltz2_affinity_constraint,
)
from proto_language.constraint.protein_structure.boltz_binding_strength_constraint import (
    BoltzBindingStrengthConfig,
    boltz_binding_strength_constraint,
)
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import SynFlowNetGenerator, SynFlowNetGeneratorConfig
from proto_language.optimizer import RejectionSamplingOptimizer, RejectionSamplingOptimizerConfig

logger = logging.getLogger(__name__)

# Example stand-in target (GB1); replace with your real target via --target-sequence.
DEFAULT_TARGET_SEQUENCE = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


def build_program(args: argparse.Namespace) -> tuple[Program, Segment]:
    """Build the multi-objective generative screen program.

    Args:
        args (argparse.Namespace): Parsed command-line options.

    Returns:
        tuple[Program, Segment]: The program and the ligand segment to read hits from.
    """
    target = Segment(sequence=args.target_sequence, sequence_type="protein", label="target")
    ligand = Segment(sequence="", sequence_type="ligand", label="candidate")
    target_construct = Construct([target])
    ligand_construct = Construct([ligand])

    generator = SynFlowNetGenerator(
        SynFlowNetGeneratorConfig(
            model_checkpoint=args.checkpoint,
            sampling_temperature=args.sampling_temperature,
            batch_size=args.num_samples,
            device=args.device,
            verbose=args.verbose,
        )
    )
    generator.assign(ligand)

    boltz_device = {"device": args.device, "verbose": int(args.verbose)}

    # Objective 1: predicted binding affinity (weighted heavier).
    affinity = Constraint(
        inputs=[target, ligand],
        function=boltz2_affinity_constraint,
        function_config=Boltz2AffinityConstraintConfig(
            return_component="binding_penalty",
            desired_affinity_value=args.desired_affinity_value,
            boltz2_affinity_config=boltz_device,
        ),
        label="Boltz2 predicted affinity",
        weight=args.affinity_weight,
    )

    # Objective 2: predicted ligand-interface confidence of the complex.
    interface_confidence = Constraint(
        inputs=[target, ligand],
        function=boltz_binding_strength_constraint,
        function_config=BoltzBindingStrengthConfig(
            return_component="ligand_iptm",
            boltz2_config=boltz_device,
        ),
        label="Boltz2 ligand-interface confidence",
        weight=args.confidence_weight,
    )

    screen = RejectionSamplingOptimizer(
        constructs=[target_construct, ligand_construct],
        generators=[generator],
        constraints=[affinity, interface_confidence],
        config=RejectionSamplingOptimizerConfig(
            num_samples=args.num_samples,
            num_results=args.num_hits,
            proposal_batch_size=args.num_samples,
            verbose=args.verbose,
            seed=args.seed,
        ),
    )

    program = Program(optimizers=[screen], num_results=args.num_hits, seed=args.seed)
    return program, ligand


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-sequence", default=DEFAULT_TARGET_SEQUENCE, help="Target protein sequence.")
    parser.add_argument("--checkpoint", default="default", help="SynFlowNet checkpoint ('default' or a .pt path).")
    parser.add_argument("--num-samples", type=int, default=64, help="Molecules to sample and score per screen.")
    parser.add_argument("--num-hits", type=int, default=8, help="Top-scoring molecules to retain.")
    parser.add_argument("--sampling-temperature", type=float, default=32.0, help="GFlowNet inverse-temperature.")
    parser.add_argument(
        "--desired-affinity-value",
        type=float,
        default=-2.0,
        help="Target predicted affinity (log10 IC50 uM); lower is stronger.",
    )
    parser.add_argument("--affinity-weight", type=float, default=2.0, help="Weight on the affinity objective.")
    parser.add_argument("--confidence-weight", type=float, default=1.0, help="Weight on the interface-confidence objective.")
    parser.add_argument("--device", default="cuda", help="Device for Boltz-2 and SynFlowNet.")
    parser.add_argument("--seed", type=int, default=0, help="Program seed for reproducibility.")
    parser.add_argument("--verbose", action="store_true", help="Verbose tool output.")
    parser.add_argument("--build-only", action="store_true", help="Build and validate the program without running it.")
    return parser.parse_args()


def main() -> None:
    """Build and run the multi-objective screen, then print the retained hits."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    program, ligand = build_program(args)
    logger.info("Built multi-objective screen: %d samples -> top %d hits", args.num_samples, args.num_hits)

    if args.build_only:
        logger.info("Build-only: program constructed and validated; skipping the run.")
        return

    program.run()

    logger.info("Top %d ligands (SMILES):", ligand.num_results)
    for rank, sequence in enumerate(ligand.result_sequences, start=1):
        logger.info("  %2d. %s", rank, sequence.sequence)


if __name__ == "__main__":
    main()
