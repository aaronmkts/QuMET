"""Configuration file loading and merging for QuMET.

This module handles loading TOML configuration files and merging configuration
values with CLI arguments according to precedence rules.
"""

import toml
from tabulate import tabulate


def convert_str_na_to_none(d):
    """Convert 'NA' strings to None in nested data structures.
    
    Since TOML does not support None values, 'NA' is used as a placeholder.
    This function recursively converts 'NA' strings back to None.
    
    Args:
        d: Data structure (dict, list, tuple, or scalar) to process.
    
    Returns:
        Processed data structure with 'NA' strings converted to None.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = convert_str_na_to_none(v)
    elif isinstance(d, list):
        d = [convert_str_na_to_none(v) for v in d]
    elif isinstance(d, tuple):
        d = tuple(convert_str_na_to_none(v) for v in d)
    else:
        if d == "NA":
            return None
        else:
            return d
    return d


def convert_none_to_str_na(d):
    """Convert None values to 'NA' strings in nested data structures.
    
    Since TOML does not support None values, this function converts None
    to 'NA' strings before saving to TOML format.
    
    Args:
        d: Data structure (dict, list, tuple, or scalar) to process.
    
    Returns:
        Processed data structure with None values converted to 'NA' strings.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = convert_none_to_str_na(v)
    elif isinstance(d, list):
        d = [convert_none_to_str_na(v) for v in d]
    elif isinstance(d, tuple):
        d = tuple(convert_none_to_str_na(v) for v in d)
    else:
        if d is None:
            return "NA"
        else:
            return d
    return d


def load_config(config_path):
    """Load configuration from a TOML file.
    
    Args:
        config_path: Path to the TOML configuration file.
    
    Returns:
        dict: Configuration dictionary with 'NA' strings converted to None.
    """
    with open(config_path) as f:
        config = toml.load(f)
    config = convert_str_na_to_none(config)
    return config


def save_config(config, config_path):
    """Save configuration to a TOML file.
    
    Args:
        config: Configuration dictionary to save.
        config_path: Path where the TOML file will be saved.
    """
    config = convert_none_to_str_na(config)
    with open(config_path, "w") as f:
        toml.dump(config, f)


def post_parse_load_config(args, defaults):
    """Load and merge configuration with CLI arguments.
    
    Merges arguments from a TOML configuration file with CLI arguments using
    the precedence rule: default < configuration < manual overrides.
    Displays a formatted table showing the effective values.
    
    Args:
        args: Parsed command-line arguments namespace.
        defaults: Dictionary of default argument values.
    
    Returns:
        argparse.Namespace: Updated arguments with merged configuration values.
    
    Raises:
        ValueError: If configuration file doesn't have .toml extension.
    """
    if args.config and not args.config.endswith(".toml"):
        raise ValueError(f"expected .toml configuration file, got {args.config}")

    # Helper function to colour the output gray
    fmt_gray = lambda x: f"\033[38;5;8m{x}\033[0m"

    fields = ["Name", "Default", "Config. File", "Manual Override", "Effective"]
    table = []

    config = load_config(args.config) if args.config else None
    for k in list(vars(args).keys()):
        if k not in defaults or k == "config":
            continue

        default_gray = fmt_gray(defaults[k])
        if config and k in config.keys():
            # Only merge the values from the configuration if there are no manual
            # overrides (i.e. the argument value doesn't deviate from its default).
            v = config[k]
            if getattr(args, k) == defaults[k]:
                setattr(args, k, v)
                table.append([k, default_gray, v, "", v])
            else:
                table.append(
                    [k, default_gray, fmt_gray(v), getattr(args, k), getattr(args, k)]
                )
        else:
            if getattr(args, k) == defaults[k]:
                table.append([k, defaults[k], "", "", defaults[k]])
            else:
                table.append([k, default_gray, "", getattr(args, k), getattr(args, k)])

    # Default values for metrics in case the [metrics] section is not present
    args.metrics_to_use = []
    args.metric_init_args = {}

    # Handle metrics from the config file, if present
    if config and "metrics" in config:
        metrics_section = config.get("metrics", {})
        # Load the list of metrics to use
        args.metrics_to_use = metrics_section.get("use_metrics", [])
        # Load initialization arguments for each metric
        for key, value in metrics_section.items():
            if key == "use_metrics":
                continue  # Skip the use_metrics list itself
            args.metric_init_args[key] = value

        # Now handle nested metrics for GMMEvaluationCallback
        if "GMMEvaluationCallback" in args.metrics_to_use:
            gmm_metric_args = args.metric_init_args.get("GMMEvaluationCallback", {})
            # For nested NDB_JSD_EvaluationCallback
            ndb_jsd_nested_args = gmm_metric_args.get(
                "NDB_JSD_EvaluationCallback", None
            )
            if ndb_jsd_nested_args is None:
                # No nested configuration provided, use global one if available
                ndb_jsd_global_args = args.metric_init_args.get(
                    "NDB_JSD_EvaluationCallback", {}
                )
                gmm_metric_args["NDB_JSD_EvaluationCallback"] = ndb_jsd_global_args
            # Else, nested configuration exists, already in gmm_metric_args
            args.metric_init_args["GMMEvaluationCallback"] = gmm_metric_args

    if not config:
        fields.remove("Config. File")
        table = [
            [k, default, override, effective]
            for k, default, _, override, effective in table
        ]

    # NOTE: We need to replace NoneType with a string 'None' for text wrapping to work
    # properly via maxcolwidths.
    table = [["None" if item is None else item for item in row] for row in table]

    print(
        tabulate(
            table,
            headers=fields,
            colalign=["left"] + ["center"] * (len(fields) - 1),
            tablefmt="pretty",
            maxheadercolwidths=24,
            maxcolwidths=24,
            disable_numparse=True,
        )
    )

    return args
