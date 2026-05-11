"""Test the PFFDTD method CLI."""
import os
import json
import pytest

from pffdtd_interface import main


# Skip the heavy end-to-end CLI test if the PFFDTD/ppffdtd solver stack is not
# importable in the test environment (e.g. minimal CI image without numba,
# pffdtd, scikit-learn, etc.). CI runners that install the full deps set will
# run the test; lean runners will skip it explicitly rather than ImportError.
def _heavy_deps_available() -> bool:
    try:
        import numba       # noqa: F401
        import scipy       # noqa: F401
        import sklearn     # noqa: F401
        # We don't require the rom/pffdtd Python sources to be on sys.path here
        # (they install via git+https from pyproject.toml), but our interface
        # module must import cleanly:
        from pffdtd_interface.pffdtd_interface import PFFDTDMethod  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _heavy_deps_available(),
    reason="PFFDTD heavy deps unavailable in this environment; CLI smoke test skipped.",
)
def test_pffdtd_method_cli(mock_requests_post, create_temporary_input_file):
    """End-to-end CLI smoke test.

    Runs `main()` which loads the input JSON, runs the PFFDTD pipeline (full FDTD
    in the test because pffdtd_train_rom = "no", pffdtd_use_rom = "no"), writes
    results back, then calls SimulationMethod.save_results().

    Asserts the canonical CHORAS schema fields are populated:
      - results[0].resultType == "PFFDTD"
      - results[0].percentage == 100
      - responses[0].receiverResults is a non-empty list (raw IR)
      - responses[0].parameters.t30 has one entry per frequency band
      - mock_requests_post called exactly once by save_results()
    """
    os.environ["JSON_PATH"] = create_temporary_input_file
    main()

    with open(create_temporary_input_file, "r") as f:
        data = json.load(f)

    r0 = data["results"][0]
    resp = r0["responses"][0]

    assert r0.get("resultType") == "PFFDTD"
    assert r0.get("percentage") == 100

    ir = resp.get("receiverResults", [])
    assert isinstance(ir, list)
    assert len(ir) > 0, "receiverResults should contain the raw IR"

    params = resp.get("parameters", {})
    n_bands = len(r0["frequencies"])
    for key in ("edt", "t20", "t30", "c80", "d50", "ts", "spl_t0_freq"):
        assert key in params, f"parameters missing '{key}'"
        assert len(params[key]) == n_bands, f"parameters[{key}] length != n_bands"

    # save_results() in the new ABC POSTs the JSON back to the executor.
    mock_requests_post.assert_called_once()


def test_pffdtd_method_cli_missing_json_path(mock_requests_post):
    """The CLI must refuse to start when JSON_PATH is unset / empty."""
    if "JSON_PATH" in os.environ:
        del os.environ["JSON_PATH"]
    with pytest.raises(FileNotFoundError, match="input_json_path cannot be None or empty"):
        main()
