"""Inference-ready generative screen: DrugFlow pocket-conditioned generation + Boltz-2 affinity.

Given a target protein (a structure plus a reference ligand that marks the pocket), this program runs
a generative screen entirely in the Proto abstractions — no model training required, only inference
against DrugFlow's public checkpoints:

- A **ligand** ``Segment`` is designed de novo by the ``DrugFlowGenerator``, which samples 3D
  molecules inside the target pocket from a pre-trained DrugFlow checkpoint.
- A **protein** ``Segment`` holds the target sequence and carries no generator.
- A ``boltz2_affinity_constraint`` wires the two into a protein-ligand complex and scores each
  generated molecule by Boltz-2's predicted binding affinity.
- A ``RejectionSamplingOptimizer`` samples a batch, scores it, and keeps the top hits.

Protein and ligand live in separate single-segment ``Construct``s; the multi-input affinity
constraint is where they meet. This is the "hand it a target, get a ranked set of posed ligands"
flow: DrugFlow proposes pocket-aware molecules and Boltz-2 ranks them by predicted binding.

Running it requires a GPU and DrugFlow's public checkpoints (auto-downloaded). Pass your real target
with ``--target-pdb`` / ``--reference-ligand`` / ``--target-sequence``; use ``--build-only`` to
construct and validate the program (with placeholder inputs) without running it.

Example:
    python examples/scripts/generative_screen_drugflow_boltz.py \
        --target-pdb target.pdb --reference-ligand ref.sdf --target-sequence "$(cat target.seq)" \
        --num-samples 32 --num-hits 8
    python examples/scripts/generative_screen_drugflow_boltz.py --build-only
"""

import argparse
import logging

from proto_language.constraint.protein_structure.boltz2_affinity_constraint import (
    Boltz2AffinityConstraintConfig,
    boltz2_affinity_constraint,
)
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import DrugFlowGenerator, DrugFlowGeneratorConfig
from proto_language.optimizer import RejectionSamplingOptimizer, RejectionSamplingOptimizerConfig

logger = logging.getLogger(__name__)

# Example target sequence: the B1 domain of streptococcal protein G (GB1), a small stand-in. Replace
# with your real target's sequence (used by the Boltz-2 affinity constraint) via --target-sequence.
DEFAULT_TARGET_SEQUENCE = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

# Minimal placeholders so --build-only can validate the program without real target files. For a real
# run, pass --target-pdb (the pocket structure) and --reference-ligand (an SDF defining the pocket).
_PLACEHOLDER_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)
_PLACEHOLDER_REF_SDF = (
    "\n     RDKit          3D\n\n"
    "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.5000    0.5000    0.5000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "M  END\n$$$$\n"
)


def build_program(args: argparse.Namespace) -> tuple[Program, Segment]:
    """Build the pocket-conditioned generative screen program.

    Args:
        args (argparse.Namespace): Parsed command-line options.

    Returns:
        tuple[Program, Segment]: The program and the ligand segment to read hits from.
    """
    target_pdb = args.target_pdb if args.target_pdb is not None else _PLACEHOLDER_PDB
    reference_ligand = args.reference_ligand if args.reference_ligand is not None else _PLACEHOLDER_REF_SDF

    # Step 1 — Segments: a fixed protein target (sequence, for affinity) and an empty ligand.
    target = Segment(sequence=args.target_sequence, sequence_type="protein", label="target")
    ligand = Segment(sequence="", sequence_type="ligand", label="candidate")

    # Step 2 — Constructs: protein and ligand are each a single-type construct.
    target_construct = Construct([target])
    ligand_construct = Construct([ligand])

    # Step 3 — Generator: DrugFlow samples pocket-conditioned molecules into the ligand segment.
    generator = DrugFlowGenerator(
        DrugFlowGeneratorConfig(
            target_structure=target_pdb,
            reference_ligand=reference_ligand,
            model_checkpoint=args.checkpoint,
            pocket_distance_cutoff=args.pocket_distance_cutoff,
            batch_size=args.num_samples,
            device=args.device,
            verbose=args.verbose,
        )
    )
    generator.assign(ligand)

    # Step 4 — Constraint: score each generated molecule by Boltz-2 predicted affinity to the target.
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
    parser.add_argument("--target-pdb", default=None, help="Target protein PDB (path or content) defining the pocket.")
    parser.add_argument("--reference-ligand", default=None, help="Reference ligand SDF (path or block) marking the pocket.")
    parser.add_argument("--target-sequence", default=DEFAULT_TARGET_SEQUENCE, help="Target protein sequence (for affinity).")
    parser.add_argument("--checkpoint", default="drugflow", help="DrugFlow checkpoint name or .ckpt path.")
    parser.add_argument("--num-samples", type=int, default=32, help="Molecules to generate and score per screen.")
    parser.add_argument("--num-hits", type=int, default=8, help="Top-scoring molecules to retain.")
    parser.add_argument("--pocket-distance-cutoff", type=float, default=8.0, help="Pocket radius (Å) around the reference.")
    parser.add_argument(
        "--desired-affinity-value",
        type=float,
        default=-2.0,
        help="Target predicted affinity (log10 IC50 uM); lower is stronger.",
    )
    parser.add_argument("--device", default="cuda", help="Device for DrugFlow and Boltz-2.")
    parser.add_argument("--seed", type=int, default=0, help="Program seed for reproducibility.")
    parser.add_argument("--verbose", action="store_true", help="Verbose tool output.")
    parser.add_argument("--build-only", action="store_true", help="Build and validate the program without running it.")
    return parser.parse_args()


def main() -> None:
    """Build and run the pocket-conditioned generative screen, then print the retained hits."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    if not args.build_only and (args.target_pdb is None or args.reference_ligand is None):
        raise SystemExit("Provide --target-pdb and --reference-ligand for a real run (or use --build-only).")

    program, ligand = build_program(args)
    logger.info("Built DrugFlow screen: %d samples -> top %d hits", args.num_samples, args.num_hits)

    if args.build_only:
        logger.info("Build-only: program constructed and validated; skipping the run.")
        return

    program.run()

    logger.info("Top %d ligands (SMILES, predicted-affinity penalty):", ligand.num_results)
    for rank, sequence in enumerate(ligand.result_sequences, start=1):
        affinity_meta = sequence.metadata.get("constraints", {}).get("Boltz2 predicted affinity", {})
        generator_meta = sequence.metadata.get("generators", {}).get("drugflow", {})
        logger.info(
            "  %2d. %s | affinity=%s | qed=%s",
            rank,
            sequence.sequence,
            affinity_meta,
            generator_meta.get("qed"),
        )


if __name__ == "__main__":
    main()
