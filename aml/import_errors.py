package_versions = {
    "seaborn": "seaborn>=0.11.2",
    "matplotlib": "matplotlib>=3.5",
}


def import_error(obj, package_name, exists):
    """
    Check if seaborn is installed.

    Returns
    -------
    bool
        True if seaborn is not installed, False otherwise.

    """

    if not exists:
        raise ImportError(
            f"{package_versions[package_name]} is required to use this function, please install it first."
        )

    return obj
