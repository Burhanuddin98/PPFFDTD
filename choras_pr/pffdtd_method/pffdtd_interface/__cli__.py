"""CLI module for PFFDTD method."""
import os
from .pffdtd_interface import PFFDTDMethod


def main() -> None:
    """Run the PFFDTD method simulation."""
    # JSON path in the uploads folder. This variable is set for the
    # container when it is started up.
    json_file_path = os.environ.get("JSON_PATH")

    print(f"Running PFFDTD method with JSON_PATH={json_file_path}")
    pffdtd_method_object = PFFDTDMethod(json_file_path)
    pffdtd_method_object.run_simulation()

    # Save the results to a separate file
    pffdtd_method_object.save_results()

    print("PFFDTD container finished.")
