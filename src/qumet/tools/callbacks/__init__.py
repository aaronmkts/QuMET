from .evaluation import (
    CosSimilarityEvaluationCallback,
    FIDEvaluationCallback,
    GMMEvaluationCallback,
    ISEvaluationCallback,
    NDB_JSD_EvaluationCallback,
    PSNRCallback,
    SSIMCallback,
)
from .visualisation import GANImagesCallback, TSNEPlotCallback


def select_callbacks(model_info, task, metrics_to_use=None, metric_init_args=None):
    """
    Selects and returns a list of callbacks based on the model type, model name, dataset information, and task.
    Optionally, it can initialize only a subset of metrics and provide initialization arguments.

    Parameters:
    - model_info (object): Contains 'model_type' and 'name' attributes.
    - dataset_info (object): Contains metadata about the dataset, such as its name.
    - task (str): The type of task being solved (e.g., 'image_generation').
    - metrics_to_use (list, optional): List of metric names to be used.
    - metric_init_args (dict, optional): Dictionary of initialization arguments for each metric callback.

    Returns:
    - List[Callback]: A list of instantiated callbacks relevant to the training configuration.

    Raises:
    - ValueError: If any selected metrics don't belong to the appropriate task for the model type.
    """
    # Mapping from class names to callback classes
    callback_classes = {
        "GANImagesCallback": GANImagesCallback,
        "FID": FIDEvaluationCallback,
        "IS": ISEvaluationCallback,
        "NDB_JSD": NDB_JSD_EvaluationCallback,
        "GMMEvaluationCallback": GMMEvaluationCallback,
        "CosineSimilarity": CosSimilarityEvaluationCallback,
        "PSNR": PSNRCallback,
        "SSIM": SSIMCallback,
        "TSNEPlotCallback": TSNEPlotCallback,
    }

    # Mapping from model types and tasks to the corresponding callbacks
    # Define the shared callbacks dictionary
    shared_callbacks = {
        "image_generation": {
            "default": [
                "GANImagesCallback",
                "FID",
                "IS",
                "NDB_JSD",
                "GMMEvaluationCallback",
                "TSNEPlotCallback",
                "CosineSimilarity",
                "PSNR",
                "SSIM",
            ],
            "apqgan": [
                "GMMEvaluationCallback",
                # You can include other metrics specific to 'apqgan' here
            ],
        },
    }

    # Define the metric_mapping with each model type as a separate key
    metric_mapping = {
        "qgan": shared_callbacks,
        "gan": shared_callbacks,
        # Add other model types as needed
    }

    callbacks = []

    model_type = model_info.model_type.value
    model_name = model_info.name.lower()
    task_lower = task.lower()

    # Check if the model type is supported
    if model_type not in metric_mapping:
        raise ValueError(f"Unsupported model type '{model_type}'.")

    # Get the task mapping
    task_mapping = metric_mapping[model_type]

    if task_lower not in task_mapping:
        raise ValueError(f"Unsupported task '{task}' for model type '{model_type}'.")

    # Get the metrics mapping for the task
    task_metrics_mapping = task_mapping[task_lower]

    # Get applicable metrics
    # Start with default metrics
    default_metrics = task_metrics_mapping.get("default", [])
    # Get model-specific metrics, if any
    model_specific_metrics = task_metrics_mapping.get(model_name, [])

    # Combine default metrics with model-specific metrics, avoiding duplicates
    applicable_metrics = default_metrics + [
        metric for metric in model_specific_metrics if metric not in default_metrics
    ]

    # If metrics_to_use is provided, filter the applicable metrics
    if metrics_to_use is not None:
        # Validate that the selected metrics belong to the task and model type
        invalid_metrics = [
            metric for metric in metrics_to_use if metric not in applicable_metrics
        ]
        if invalid_metrics:
            raise ValueError(
                f"Metrics {invalid_metrics} are not valid for the task '{task}' with model '{model_name}'."
            )
        # Use the metrics in metrics_to_use, preserving the order specified
        applicable_metrics = [
            metric for metric in metrics_to_use if metric in applicable_metrics
        ]
    else:
        # If metrics_to_use is None, use the applicable_metrics as is
        pass

    # Initialize each callback, passing any initialization arguments
    for metric in applicable_metrics:
        # Get initialization args if provided, otherwise default to an empty dict
        init_args = metric_init_args.get(metric, {}) if metric_init_args else {}

        # Special handling for GMMEvaluationCallback to pass nested callback args
        if metric == "GMMEvaluationCallback":
            # Extract nested args for internal callbacks
            ndb_jsd_args = init_args.get("NDB_JSD_EvaluationCallback", {})
            gan_images_args = init_args.get("GANImagesCallback", {})

            # Remove nested args from init_args to avoid conflicts
            init_args = {
                k: v
                for k, v in init_args.items()
                if k not in ["NDB_JSD_EvaluationCallback", "GANImagesCallback"]
            }

            # Instantiate GMMEvaluationCallback with nested args
            callbacks.append(
                callback_classes[metric](
                    **init_args,
                    ndb_jsd_args=ndb_jsd_args,
                    gan_images_args=gan_images_args,
                )
            )
        else:
            # Instantiate the callback with the provided initialization arguments
            callbacks.append(callback_classes[metric](**init_args))

    return callbacks
