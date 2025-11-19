#!/usr/bin/env python3
"""
SGD Analysis Wizard - Interactive Configuration and Execution

This script guides users through setting up and running SGD detection analysis
with an easy-to-use question/answer interface. It can save configurations for
reuse and run the analysis automatically.

Usage:
    python sgd_wizard.py                    # Interactive mode
    python sgd_wizard.py --config myconfig.json  # Use saved config
    python sgd_wizard.py --save-only        # Create config without running
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}\n")

def print_section(text):
    """Print a section header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}▶ {text}{Colors.ENDC}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

def ask_question(question, default=None, options=None, validation=None):
    """
    Ask user a question and return the answer

    Args:
        question: The question to ask
        default: Default value if user just presses Enter
        options: List of valid options (if applicable)
        validation: Function to validate input
    """
    if options:
        print(f"{Colors.BOLD}{question}{Colors.ENDC}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        if default:
            prompt = f"Choice [default: {default}]: "
        else:
            prompt = "Choice: "
    else:
        if default:
            prompt = f"{Colors.BOLD}{question}{Colors.ENDC} [{default}]: "
        else:
            prompt = f"{Colors.BOLD}{question}{Colors.ENDC}: "

    while True:
        answer = input(prompt).strip()

        # Use default if provided and user pressed Enter
        if not answer and default is not None:
            return default

        # Handle options
        if options:
            try:
                choice = int(answer)
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    print_error(f"Please enter a number between 1 and {len(options)}")
                    continue
            except ValueError:
                # Check if user typed the option name directly
                if answer in options:
                    return answer
                print_error("Please enter a valid number")
                continue

        # Validate if validation function provided
        if validation:
            is_valid, message = validation(answer)
            if not is_valid:
                print_error(message)
                continue

        return answer

def validate_path_exists(path):
    """Validate that a path exists"""
    if Path(path).exists():
        return True, ""
    return False, f"Path does not exist: {path}"

def validate_positive_number(value):
    """Validate that a value is a positive number"""
    try:
        num = float(value)
        if num > 0:
            return True, ""
        return False, "Value must be positive"
    except ValueError:
        return False, "Please enter a valid number"

def validate_yes_no(value):
    """Validate yes/no input"""
    if value.lower() in ['y', 'yes', 'n', 'no']:
        return True, ""
    return False, "Please enter 'y' or 'n'"

def check_sam_installed():
    """Check if SAM is installed and available"""
    try:
        # Try to import SAM
        import segment_anything

        # Check if checkpoint exists
        models_dir = Path("models/sam")
        if models_dir.exists():
            pth_files = list(models_dir.glob("*.pth"))
            if pth_files:
                return True

        # SAM installed but no checkpoint
        return False
    except ImportError:
        return False

def get_configuration_interactive():
    """Get configuration through interactive questions"""

    print_header("SGD Detection Analysis Wizard")
    print_info("This wizard will guide you through configuring your SGD detection analysis.")
    print_info("Press Enter to accept default values shown in [brackets].\n")

    config = {}

    # ===== Data Input =====
    print_section("1. Data Input")

    config['data_dir'] = ask_question(
        "Path to directory containing thermal images (e.g., data/100MEDIA)",
        default="data/100MEDIA",
        validation=validate_path_exists
    )

    # Check what files are in the directory
    data_path = Path(config['data_dir'])
    rgb_files = list(data_path.glob("MAX_*.JPG"))
    thermal_files = list(data_path.glob("IRX_*.irg"))

    print_success(f"Found {len(rgb_files)} RGB images and {len(thermal_files)} thermal images")

    if len(rgb_files) == 0 or len(thermal_files) == 0:
        print_warning("Warning: Missing RGB or thermal files. Analysis may fail.")

    # ===== Segmentation Training (Optional) =====
    print_section("2. Segmentation Model Setup (Optional)")

    print_info("Ocean/land segmentation is critical for accurate SGD detection")
    print_info("Two approaches available:")
    print_info("  1. Random Forest - CPU-based, needs training per environment")
    print_info("  2. SAM (GPU) - Advanced, zero-shot, better boundaries")
    print_info("  3. Skip - Use existing model or default settings\n")

    seg_choice = ask_question(
        "Choose segmentation approach",
        options=["Random Forest (train new model)", "SAM (create prompts - requires GPU)", "Skip (use existing)"],
        default="3"
    )

    # Store segmentation type in config
    config['segmentation_type'] = seg_choice

    if "Random Forest" in seg_choice:
        # Original Random Forest training flow
        print_info("\nLaunching Random Forest segmentation trainer...")
        print_info("Instructions:")
        print_info("  - Press 1 to label ocean (blue)")
        print_info("  - Press 2 to label land (green)")
        print_info("  - Press 3 to label rock (gray)")
        print_info("  - Press 4 to label waves (white)")
        print_info("  - Press N for next frame, P for previous")
        print_info("  - Press R for random frame")
        print_info("  - Press ] to skip +5, } to skip +10")
        print_info("  - Press T to train model")
        print_info("  - Press S to save model")
        print_info("  - Press Q to quit trainer\n")

        proceed = ask_question(
            "Launch Random Forest trainer now?",
            default="y",
            validation=validate_yes_no
        ).lower() in ['y', 'yes']

        if proceed:
            # Launch the segmentation trainer
            trainer_path = Path(__file__).parent / "train_segmentation.py"
            try:
                result = subprocess.run(
                    [sys.executable, str(trainer_path), "--data", config['data_dir']],
                    check=False
                )
                if result.returncode == 0:
                    print_success("Random Forest trainer completed successfully!")
                    # List available models
                    models_dir = Path("models")
                    if models_dir.exists():
                        model_files = sorted(models_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
                        if model_files:
                            print_success(f"Found {len(model_files)} trained models:")
                            for i, model in enumerate(model_files[:5], 1):
                                print(f"  {i}. {model.name}")
                else:
                    print_warning("Trainer exited. Continuing with wizard...")
            except FileNotFoundError:
                print_error(f"Could not find trainer at {trainer_path}")
                print_info("Continuing with wizard...")
        else:
            print_info("Skipping training. You can train later with:")
            print_info(f"  python scripts/train_segmentation.py --data {config['data_dir']}")

    elif "SAM" in seg_choice:
        # SAM setup flow
        print_info("\nSAM (Segment Anything Model) Setup")
        print_info("SAM provides superior segmentation with GPU acceleration")

        # Check if SAM is installed
        sam_available = check_sam_installed()

        if not sam_available:
            print_warning("SAM is not installed!")
            install_sam = ask_question(
                "Install SAM now? (requires GPU)",
                default="y",
                validation=validate_yes_no
            ).lower() in ['y', 'yes']

            if install_sam:
                print_info("\nRunning SAM installation script...")
                print_info("You'll be asked to choose model size:")
                print_info("  1. ViT-H (2.5GB) - Best accuracy for DGX/workstations")
                print_info("  2. ViT-L (1.2GB) - Good accuracy, faster")
                print_info("  3. ViT-B (375MB) - Fastest, good for testing\n")

                setup_path = Path(__file__).parent / "setup_sam.sh"
                try:
                    result = subprocess.run(["bash", str(setup_path)], check=False)
                    if result.returncode == 0:
                        print_success("SAM installed successfully!")
                        sam_available = True
                    else:
                        print_error("SAM installation failed")
                        print_info("Continuing with wizard...")
                except Exception as e:
                    print_error(f"Installation error: {e}")
                    print_info("Continuing with wizard...")
            else:
                print_info("Skipping SAM installation")
                print_info("You can install later with: bash scripts/setup_sam.sh")

        if sam_available:
            # Create prompts using the new streamlined tool
            print_info("\nSAM Prompt Creator & Batch Processor")
            print_info("")
            print_info("Simple workflow:")
            print_info("  1. Left-click ocean areas (blue dots)")
            print_info("  2. Right-click land areas to exclude (red X's)")
            print_info("  3. Press W to save prompts")
            print_info("  4. Press → to test on more images (optional)")
            print_info("  5. Press P to process ALL images → DONE!")
            print_info("")
            print_warning("SAM prompts are image-specific - one set per flight usually works!")
            print_info("")

            create_prompts = ask_question(
                "Create SAM prompts and process images now?",
                default="y",
                validation=validate_yes_no
            ).lower() in ['y', 'yes']

            if create_prompts:

                # Find a sample image
                data_path = Path(config['data_dir'])
                rgb_images = list(data_path.glob("MAX_*.JPG"))
                if not rgb_images:
                    print_error(f"No MAX_*.JPG images found in {config['data_dir']}")
                else:
                    sample_image = str(rgb_images[len(rgb_images)//2])  # Use middle image
                    print_info(f"Using sample image: {Path(sample_image).name}")

                    creator_path = Path(__file__).parent / "sam_prompt_creator.py"
                    try:
                        result = subprocess.run(
                            f'{sys.executable} {str(creator_path)} --image "{sample_image}"',
                            shell=True,
                            check=False
                        )
                        if result.returncode == 0:
                            print_success("SAM prompt creator closed")
                            # List available prompts
                            prompts_dir = Path("prompts")
                            if prompts_dir.exists():
                                prompt_files = sorted(prompts_dir.glob("sam_*.json"),
                                                    key=lambda p: p.stat().st_mtime, reverse=True)
                                if prompt_files:
                                    print_success(f"Found {len(prompt_files)} prompt files:")
                                    for i, pf in enumerate(prompt_files[:5], 1):
                                        print(f"  {i}. {pf.name}")

                                    # Ask which to use
                                    config['sam_prompts'] = str(prompt_files[0])
                                    print_info(f"Will use: {prompt_files[0].name}")
                        else:
                            print_warning("Prompt creator exited. Continuing with wizard...")
                    except Exception as e:
                        print_error(f"Error launching comparison tool: {e}")
                        print_info("Continuing with wizard...")
            else:
                print_info("Skipping prompt creation")
                # Check for existing prompts
                prompts_dir = Path("prompts")
                if prompts_dir.exists():
                    prompt_files = list(prompts_dir.glob("*.json"))
                    if prompt_files:
                        print_info(f"\nFound {len(prompt_files)} existing prompt files")
                        use_existing = ask_question(
                            "Use existing prompts?",
                            default="y",
                            validation=validate_yes_no
                        ).lower() in ['y', 'yes']

                        if use_existing:
                            print("\nAvailable prompts:")
                            for i, pf in enumerate(prompt_files, 1):
                                print(f"  {i}. {pf.name}")

                            prompt_choice = ask_question(
                                "Which prompts to use? (enter number)",
                                default="1"
                            )
                            try:
                                idx = int(prompt_choice) - 1
                                if 0 <= idx < len(prompt_files):
                                    config['sam_prompts'] = str(prompt_files[idx])
                                    print_success(f"Will use: {prompt_files[idx].name}")
                            except ValueError:
                                print_warning("Invalid choice, skipping prompts")

    else:
        # Skip - use existing
        print_info("Using existing segmentation model/settings")
        print_info("Default Random Forest model will be used if available")

    # ===== Output Configuration =====
    print_section("3. Output Configuration")

    # Suggest output name based on input directory
    suggested_name = data_path.name.lower().replace('media', '')
    if not suggested_name:
        suggested_name = "sgd_results"

    config['output_name'] = ask_question(
        "Output file name (without extension)",
        default=suggested_name
    )

    # Output directory
    config['output_dir'] = ask_question(
        "Output directory",
        default="sgd_output"
    )

    # ===== Detection Parameters =====
    print_section("4. Detection Parameters")

    # Choose detection method
    print_info("Detection method: Choose how to identify SGD features")
    detection_method_options = [
        "threshold - Automatic temperature-based detection (fast, for batch processing)",
        "sam - SAM-based interactive detection (more accurate, manual review)"
    ]

    detection_method = ask_question(
        "Select detection method",
        default=detection_method_options[0],
        options=detection_method_options
    )

    config['detection_method'] = detection_method.split(' - ')[0]

    if config['detection_method'] == 'threshold':
        print_info("Temperature threshold: How much cooler (°C) than ocean to detect as SGD")
        config['temp_threshold'] = float(ask_question(
            "Temperature threshold (°C)",
            default="0.5",
            validation=validate_positive_number
        ))

        print_info("Minimum area: Smallest plume size to detect (pixels)")
        config['min_area'] = int(ask_question(
            "Minimum plume area (pixels)",
            default="50",
            validation=validate_positive_number
        ))
    else:
        # SAM-based detection - use defaults for now
        config['temp_threshold'] = 0.5
        config['min_area'] = 50
        print_info("SAM detection uses interactive feature selection")

    # ===== Detector Type =====
    print_section("5. Detector Configuration")

    detector_options = [
        "integrated - Standard detector (fastest)",
        "improved - Enhanced baseline methods + sun glint filtering (recommended)",
        "temporal - Moving average smoothing (for video sequences)",
        "edge_aware - Frame boundary handling (for overlapping frames)"
    ]

    detector_choice = ask_question(
        "Select detector type",
        default=detector_options[1],
        options=detector_options
    )
    config['detector'] = detector_choice.split(' - ')[0]

    # ===== Baseline Method (for improved detector) =====
    if config['detector'] == 'improved':
        print_section("6. Baseline Method")
        print_info("Baseline method: How to calculate the ocean background temperature")

        baseline_options = [
            "upper_quartile - 75th percentile (recommended, robust to cold plumes)",
            "median - Middle value (traditional)",
            "trimmed_mean - Mean after removing lowest 25%",
            "modal_peak - Most common temperature"
        ]

        baseline_choice = ask_question(
            "Select baseline method",
            default=baseline_options[0],
            options=baseline_options
        )
        config['baseline_method'] = baseline_choice.split(' - ')[0]

    # ===== ML Segmentation =====
    # Only ask about ML if user chose "Skip" in section 2
    # If they chose SAM or Random Forest, don't ask again
    seg_type = config.get('segmentation_type', '')

    if 'SAM' in seg_type or 'Random Forest' in seg_type:
        # User already configured segmentation, skip this section
        print_info("ℹ Skipping ML configuration (already configured segmentation in section 2)")
        config['use_ml'] = False
    else:
        # User chose "Skip" - ask about ML segmentation
        print_section("7. Machine Learning Segmentation")

        # Check for available models
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl"))
            if model_files:
                print_success(f"Found {len(model_files)} trained models")
                print_info("Available models:")
                for model in model_files:
                    print(f"  - {model.name}")

        use_ml = ask_question(
            "Use ML segmentation? (y/n)",
            default="y",
            validation=validate_yes_no
        ).lower() in ['y', 'yes']

        config['use_ml'] = use_ml

        if use_ml:
            config['ml_model'] = ask_question(
                "ML model file name",
                default="segmentation_model.pkl"
            )

    # ===== Advanced Options =====
    print_section("8. Advanced Options")

    show_advanced = ask_question(
        "Configure advanced options? (y/n)",
        default="n",
        validation=validate_yes_no
    ).lower() in ['y', 'yes']

    if show_advanced:
        config['detect_glint'] = ask_question(
            "Enable sun glint detection? (y/n)",
            default="y",
            validation=validate_yes_no
        ).lower() in ['y', 'yes']

        config['min_shore_distance'] = int(ask_question(
            "Minimum distance from shore (pixels)",
            default="5",
            validation=validate_positive_number
        ))

        config['max_shore_distance'] = int(ask_question(
            "Maximum distance from shore (pixels)",
            default="200",
            validation=validate_positive_number
        ))
    else:
        # Use defaults
        config['detect_glint'] = True
        config['min_shore_distance'] = 5
        config['max_shore_distance'] = 200

    # ===== Output Formats =====
    print_section("9. Output Formats")

    config['export_kml'] = ask_question(
        "Export KML (Google Earth)? (y/n)",
        default="y",
        validation=validate_yes_no
    ).lower() in ['y', 'yes']

    config['export_geojson'] = ask_question(
        "Export GeoJSON? (y/n)",
        default="y",
        validation=validate_yes_no
    ).lower() in ['y', 'yes']

    config['export_csv'] = ask_question(
        "Export CSV? (y/n)",
        default="y",
        validation=validate_yes_no
    ).lower() in ['y', 'yes']

    return config

def display_configuration(config):
    """Display the configuration summary"""
    print_section("Configuration Summary")

    print(f"{Colors.BOLD}Data Input:{Colors.ENDC}")
    print(f"  Data Directory: {config['data_dir']}")

    print(f"\n{Colors.BOLD}Output:{Colors.ENDC}")
    print(f"  Output Name: {config['output_name']}")
    print(f"  Output Directory: {config['output_dir']}")

    print(f"\n{Colors.BOLD}Detection Parameters:{Colors.ENDC}")
    print(f"  Detection Method: {config.get('detection_method', 'threshold').upper()}")
    print(f"  Temperature Threshold: {config['temp_threshold']}°C")
    print(f"  Minimum Area: {config['min_area']} pixels")
    print(f"  Detector: {config['detector']}")
    if 'baseline_method' in config:
        print(f"  Baseline Method: {config['baseline_method']}")

    print(f"\n{Colors.BOLD}Segmentation:{Colors.ENDC}")
    print(f"  Use ML: {config['use_ml']}")
    if config['use_ml']:
        print(f"  Model: {config['ml_model']}")

    print(f"\n{Colors.BOLD}Output Formats:{Colors.ENDC}")
    formats = []
    if config['export_kml']:
        formats.append("KML")
    if config['export_geojson']:
        formats.append("GeoJSON")
    if config['export_csv']:
        formats.append("CSV")
    print(f"  {', '.join(formats)}")
    print()

def save_configuration(config, filepath):
    """Save configuration to JSON file"""
    # Add metadata
    config['created_at'] = datetime.now().isoformat()
    config['version'] = '1.0'

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    print_success(f"Configuration saved to: {filepath}")

def load_configuration(filepath):
    """Load configuration from JSON file"""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        print_success(f"Configuration loaded from: {filepath}")
        return config
    except FileNotFoundError:
        print_error(f"Configuration file not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print_error(f"Invalid JSON in configuration file: {filepath}")
        sys.exit(1)

def run_sam_detection(config):
    """Run SAM-based interactive SGD detection"""

    print_section("SAM-Based SGD Detection")

    print_info("SAM detection provides an interactive workflow for precise SGD identification.")
    print_info("You'll review images one-by-one, clicking on cold spots to identify SGD features.")
    print()

    # Find RGB and thermal pairs
    data_dir = Path(config['data_dir'])
    rgb_files = sorted(data_dir.glob("MAX_*.JPG"))

    if not rgb_files:
        print_error(f"No RGB images (MAX_*.JPG) found in {data_dir}")
        return False

    print_info(f"Found {len(rgb_files)} images to process")
    print()

    # Guide user through SAM workflow
    print(f"{Colors.BOLD}SAM Detection Workflow:{Colors.ENDC}")
    print("  1. The interactive tool will open showing thermal data")
    print("  2. Look at the thermal deviation map (top left)")
    print("  3. Click on cold spots (blue areas) to mark potential SGD")
    print("  4. Press 'S' to run SAM segmentation")
    print("  5. Compare SAM results (red) with threshold method (green)")
    print("  6. Press 'Q' when done reviewing")
    print()

    response = input(f"{Colors.CYAN}Ready to start SAM detection? (y/n): {Colors.ENDC}")
    if response.lower() != 'y':
        print_info("SAM detection cancelled")
        return False

    # Launch interactive tool on first image
    rgb_path = rgb_files[0]
    thermal_path = data_dir / rgb_path.name.replace("MAX_", "IRX_").replace(".JPG", ".irg")

    if not thermal_path.exists():
        print_error(f"Thermal file not found: {thermal_path}")
        return False

    # Check if ocean mask exists
    output_dir = Path(config['output_dir']) / config['output_name']
    ocean_mask_dir = output_dir / "ocean_masks"
    ocean_mask_path = ocean_mask_dir / f"{rgb_path.stem}_ocean.npy"

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "test_sam_sgd_detection.py"),
        "--rgb", str(rgb_path),
        "--thermal", str(thermal_path)
    ]

    if ocean_mask_path.exists():
        cmd.extend(["--ocean-mask", str(ocean_mask_path)])
        print_info(f"Using existing ocean mask: {ocean_mask_path.name}")
    else:
        print_info("Using automatic ocean detection (basic color threshold)")

    print()
    print_info("Launching SAM detection tool...")
    print(f"{Colors.CYAN}{' '.join(cmd)}{Colors.ENDC}\n")

    try:
        subprocess.run(cmd, check=True)
        print()
        print_success("SAM detection tool closed")

        # Offer to process more images
        print()
        print_info("SAM detection is currently an interactive tool for manual review.")
        print_info("To process more images, run the tool again on different image pairs.")
        print()
        print(f"{Colors.BOLD}Next steps:{Colors.ENDC}")
        print(f"  • Review your findings")
        print(f"  • For batch processing, use threshold-based detection instead")
        print(f"  • Or process each image manually with:")
        print(f"    python scripts/test_sam_sgd_detection.py --rgb <RGB> --thermal <THERMAL>")
        print()

        return True

    except subprocess.CalledProcessError as e:
        print_error(f"\nSAM detection failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print_error("\nCould not find test_sam_sgd_detection.py")
        print_info("Make sure you're running this script from the correct directory")
        return False
    except KeyboardInterrupt:
        print()
        print_info("SAM detection interrupted by user")
        return False

def run_analysis(config):
    """Run the SGD analysis with the given configuration"""

    print_section("Running Analysis")

    # Handle SAM-based detection differently
    if config.get('detection_method') == 'sam':
        return run_sam_detection(config)

    # Build command for threshold-based detection
    script_path = Path(__file__).parent / "sgd_autodetect.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--data", config['data_dir'],
        "--output", str(Path(config['output_dir']) / config['output_name']),
        "--temp", str(config['temp_threshold']),  # Fixed: was --temp-threshold
        "--area", str(config['min_area']),        # Fixed: was --min-area
    ]

    # Add detector type
    if config['detector'] == 'improved':
        cmd.extend(["--detector", "improved"])
        if 'baseline_method' in config:
            cmd.extend(["--baseline-method", config['baseline_method']])

    # Add ML options
    if config['use_ml'] and 'ml_model' in config:
        cmd.extend(["--model", config['ml_model']])
    # Note: sgd_autodetect.py doesn't have --no-ml flag, it just won't use ML if no --model

    # Add advanced options
    if config.get('detect_glint', True):
        cmd.append("--filter-glint")  # Fixed: was --detect-glint

    print_info(f"Running command:")
    print(f"{Colors.CYAN}{' '.join(cmd)}{Colors.ENDC}\n")

    # Run the analysis
    try:
        result = subprocess.run(cmd, check=True)
        print_success("\nAnalysis completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"\nAnalysis failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print_error(f"\nCould not find sgd_autodetect.py at {script_path}")
        print_info("Make sure you're running this script from the correct directory")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Interactive wizard for SGD detection analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - wizard guides you through setup
  python sgd_wizard.py

  # Use a saved configuration
  python sgd_wizard.py --config my_analysis.json

  # Create config without running analysis
  python sgd_wizard.py --save-only --output my_config.json

  # Load config, modify data directory, and run
  python sgd_wizard.py --config base_config.json --data new_data/101MEDIA
        """
    )

    parser.add_argument('--config', help='Load configuration from JSON file')
    parser.add_argument('--save-only', action='store_true',
                       help='Create and save configuration without running analysis')
    parser.add_argument('--output', default='sgd_config.json',
                       help='Output filename for saved configuration (default: sgd_config.json)')
    parser.add_argument('--data', help='Override data directory from config')
    parser.add_argument('--no-confirm', action='store_true',
                       help='Skip confirmation prompt before running')

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = load_configuration(args.config)
        display_configuration(config)
    else:
        config = get_configuration_interactive()
        display_configuration(config)

    # Override data directory if provided
    if args.data:
        config['data_dir'] = args.data
        print_info(f"Data directory overridden: {args.data}")

    # Save configuration
    save_configuration(config, args.output)

    # Run analysis unless --save-only
    if not args.save_only:
        if not args.no_confirm:
            print()
            proceed = ask_question(
                "Proceed with analysis?",
                default="y",
                validation=validate_yes_no
            )
            if proceed.lower() not in ['y', 'yes']:
                print_info("Analysis cancelled.")
                return

        success = run_analysis(config)

        if success:
            print_section("Next Steps")
            print_info(f"1. Check results in: {config['output_dir']}/")
            print_info(f"2. Open KML files in Google Earth for visualization")
            print_info(f"3. Reuse this configuration: python sgd_wizard.py --config {args.output}")
            print_info(f"4. Run on different data: python sgd_wizard.py --config {args.output} --data path/to/data")
    else:
        print_info(f"\nConfiguration saved. Run analysis with:")
        print(f"{Colors.CYAN}python sgd_wizard.py --config {args.output}{Colors.ENDC}")

if __name__ == "__main__":
    main()
