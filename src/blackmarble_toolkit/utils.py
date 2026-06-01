import os
import ee


def initialize_ee(project_name: str | None = None) -> None:
    """
    Attempts to initialize Earth Engine. Falls back to authentication if required.

    Args:
        project_name: Optional explicit project name. If not provided,
            it attempts to read the 'GOOGLE_CLOUD_PROJECT' environment variable.
    """
    project = project_name or os.environ.get("GOOGLE_CLOUD_PROJECT")

    if not project:
        raise ValueError(
            "Earth Engine project name not provided. Pass it to the function "
            "or set the 'GOOGLE_CLOUD_PROJECT' environment variable."
        )

    try:
        ee.Initialize(project=project)
    except ee.ee_exception.EEException:
        ee.Authenticate()
        ee.Initialize(project=project)
