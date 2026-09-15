"""
Main execution file for the Hierarchical Factor Graph Optimization system.

Launches the GUI calibration tool.

Usage:
    cd hierarchical_fgo
    python main.py
"""

import sys
import os

# Add the current working directory (hierarchical_fgo) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def main():
    """Launch the GUI calibration tool."""

    print("\n" + "=" * 60)
    print("Hierarchical Factor Graph Optimization - GUI Calibration Tool")
    print("=" * 60)
    print("[gui] Launching graphical interface...")
    print("=" * 60 + "\n")

    try:
        from PySide6.QtWidgets import QApplication
        from calibration_tool import PathSelector

        app = QApplication(sys.argv)
        window = PathSelector()
        window.show()
        exit_code = app.exec()

        # Leave without unwinding: interpreter teardown destroys Qt widgets,
        # matplotlib canvases, gtsam values and the AprilTag detector's worker
        # threads in Python's arbitrary GC order, which crashes the process
        # after a calibration has run. See PathSelector.closeEvent.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    except ImportError as e:
        print("❌ PySide6 is required to run the GUI but is not installed.")
        print("  Install command: pip install PySide6")
        print(f"\n  Error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Error while running the GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠  Program interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
