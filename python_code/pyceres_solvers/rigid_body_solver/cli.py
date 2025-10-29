"""Command-line interface for rigid body tracking."""

import argparse
from pathlib import Path
import sys
import logging

from core.topology import RigidBodyTopology
from core.optimization import OptimizationConfig
from api import TrackingConfig, process_tracking_data, inspect_csv, create_topology_from_csv

logger = logging.getLogger(__name__)


def cmd_inspect(*, args: argparse.Namespace) -> None:
    """Inspect CSV file and show available markers."""
    inspect_csv(csv_path=Path(args.input))


def cmd_process(*, args: argparse.Namespace) -> None:
    """Process tracking data."""

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Load or create topology
    if args.topology:
        logger.info(f"Loading topology from: {args.topology}")
        topology = RigidBodyTopology.load_json(filepath=Path(args.topology))
    elif args.markers:
        logger.info("Creating topology from specified markers...")
        marker_list = [m.strip() for m in args.markers.split(',')]
        topology = RigidBodyTopology.from_marker_names(
            marker_names=marker_list,
            edge_strategy=args.edge_strategy,
            name=args.name or "custom"
        )
    else:
        logger.info("Creating topology from all markers in CSV...")
        topology = create_topology_from_csv(
            csv_path=input_path,
            edge_strategy=args.edge_strategy,
            name=args.name or "auto"
        )

    # Create configuration
    config = TrackingConfig(
        input_csv=input_path,
        topology=topology,
        output_dir=output_dir,
        scale_factor=args.scale,
        z_value=args.z_value,
        likelihood_threshold=args.likelihood_threshold,
        csv_format=args.format,
        reference_method=args.reference_method,
        optimization=OptimizationConfig(
            max_iter=args.max_iter,
            lambda_data=args.lambda_data,
            lambda_rigid=args.lambda_rigid,
            lambda_rot_smooth=args.lambda_rot_smooth,
            lambda_trans_smooth=args.lambda_trans_smooth
        ),
        copy_viewer=not args.no_viewer
    )

    # Process
    process_tracking_data(config=config)


def cmd_create_topology(*, args: argparse.Namespace) -> None:
    """Create and save topology file."""

    marker_list = [m.strip() for m in args.markers.split(',')]

    topology = RigidBodyTopology.from_marker_names(
        marker_names=marker_list,
        edge_strategy=args.edge_strategy,
        name=args.name or "custom"
    )

    output_path = Path(args.output)
    topology.save_json(filepath=output_path)

    logger.info(f"\n✓ Topology saved to: {output_path}")
    logger.info(f"  Markers: {len(topology.marker_names)}")
    logger.info(f"  Edges: {len(topology.rigid_edges)}")


def main() -> None:
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Rigid Body Tracking with PyCeres",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Inspect CSV file
  python cli.py inspect data.csv

  # Process with auto-generated topology
  python cli.py process data.csv --output results/

  # Process with specific markers
  python cli.py process data.csv --output results/ \\
      --markers "nose,left_ear,right_ear,base"

  # Process with custom topology file
  python cli.py process data.csv --output results/ \\
      --topology my_topology.json

  # Create topology file
  python cli.py create-topology \\
      --markers "nose,left_ear,right_ear" \\
      --output topology.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # ========== INSPECT command ==========
    inspect_parser = subparsers.add_parser(
        'inspect',
        help='Inspect CSV file and show available markers'
    )
    inspect_parser.add_argument(
        'input',
        help='Input CSV file'
    )

    # ========== PROCESS command ==========
    process_parser = subparsers.add_parser(
        'process',
        help='Process tracking data'
    )

    # Input/output
    process_parser.add_argument(
        'input',
        help='Input CSV file with trajectory data'
    )
    process_parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output directory (default: output/)'
    )

    # Topology options (mutually exclusive)
    topology_group = process_parser.add_mutually_exclusive_group()
    topology_group.add_argument(
        '--topology', '-t',
        help='Path to topology JSON file'
    )
    topology_group.add_argument(
        '--markers', '-m',
        help='Comma-separated list of marker names (e.g., "nose,left_ear,right_ear")'
    )

    process_parser.add_argument(
        '--edge-strategy',
        choices=['full', 'minimal', 'skeleton'],
        default='full',
        help='Edge connection strategy (default: full)'
    )
    process_parser.add_argument(
        '--name',
        help='Name for topology (default: auto-generated)'
    )

    # Data loading options
    process_parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Scale factor for coordinates (default: 1.0, use 0.001 for mm to m)'
    )
    process_parser.add_argument(
        '--z-value',
        type=float,
        default=0.0,
        help='Default z-coordinate for 2D data (default: 0.0)'
    )
    process_parser.add_argument(
        '--likelihood-threshold',
        type=float,
        help='For DLC data, filter points below this confidence'
    )
    process_parser.add_argument(
        '--format',
        choices=['tidy', 'wide', 'dlc'],
        help='Force specific CSV format (auto-detected if not specified)'
    )

    # Reference estimation
    process_parser.add_argument(
        '--reference-method',
        choices=['median', 'mean'],
        default='median',
        help='Reference geometry estimation method (default: median)'
    )

    # Optimization weights
    process_parser.add_argument(
        '--max-iter',
        type=int,
        default=300,
        help='Maximum optimization iterations (default: 300)'
    )
    process_parser.add_argument(
        '--lambda-data',
        type=float,
        default=100.0,
        help='Data fitting weight (default: 100.0)'
    )
    process_parser.add_argument(
        '--lambda-rigid',
        type=float,
        default=500.0,
        help='Rigid constraint weight (default: 500.0)'
    )
    process_parser.add_argument(
        '--lambda-rot-smooth',
        type=float,
        default=200.0,
        help='Rotation smoothness weight (default: 200.0)'
    )
    process_parser.add_argument(
        '--lambda-trans-smooth',
        type=float,
        default=200.0,
        help='Translation smoothness weight (default: 200.0)'
    )

    # Output options
    process_parser.add_argument(
        '--no-viewer',
        action='store_true',
        help='Do not copy viewer HTML to output directory'
    )

    # ========== CREATE-TOPOLOGY command ==========
    create_parser = subparsers.add_parser(
        'create-topology',
        help='Create and save topology JSON file'
    )
    create_parser.add_argument(
        '--markers', '-m',
        required=True,
        help='Comma-separated list of marker names'
    )
    create_parser.add_argument(
        '--output', '-o',
        default='topology.json',
        help='Output JSON file (default: topology.json)'
    )
    create_parser.add_argument(
        '--edge-strategy',
        choices=['full', 'minimal', 'skeleton'],
        default='full',
        help='Edge connection strategy (default: full)'
    )
    create_parser.add_argument(
        '--name',
        help='Name for topology (default: custom)'
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )

    # Execute command
    if args.command == 'inspect':
        cmd_inspect(args=args)
    elif args.command == 'process':
        cmd_process(args=args)
    elif args.command == 'create-topology':
        cmd_create_topology(args=args)


if __name__ == "__main__":
    main()