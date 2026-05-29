import os
import ee


def initialize_ee(project_name: str | None = None) -> None:
    """
    Attempts to initialize Earth Engine. Falls back to authentication if required.

    Args:
        project_name: The GCP project name for EE initialization. If None,
            it attempts to read the 'EE_PROJECT' environment variable.
    """
    project = project_name or os.environ.get("EE_PROJECT")

    if not project:
        raise ValueError(
            "Earth Engine project name not provided. Pass it to the function "
            "or set the 'EE_PROJECT' environment variable."
        )

    try:
        ee.Initialize(project=project)
    except ee.ee_exception.EEException:
        ee.Authenticate()
        ee.Initialize(project=project)
