from typing import TYPE_CHECKING, Any, Callable, List, Union, cast

import torch
from torch import Tensor
from typing_extensions import Literal

from transformers import CLIPModel as _CLIPModel
from transformers import CLIPProcessor as _CLIPProcessor

from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_10


def _get_clip_model_and_processor(
    model_name_or_path: Union[
        Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14",
            "jinaai/jina-clip-v2",
            "zer0int/LongCLIP-L-Diffusers",
            "zer0int/LongCLIP-GmP-ViT-L-14",
        ],
        Callable[[], tuple[_CLIPModel, _CLIPProcessor]],
    ],
) -> tuple[_CLIPModel, _CLIPProcessor]:
    if callable(model_name_or_path):
        return model_name_or_path()

    if _TRANSFORMERS_GREATER_EQUAL_4_10:
        from transformers import AutoModel, AutoProcessor
        from transformers import CLIPConfig as _CLIPConfig
        from transformers import CLIPModel as _CLIPModel
        from transformers import CLIPProcessor as _CLIPProcessor

        if "openai" in model_name_or_path:
            model = _CLIPModel.from_pretrained(model_name_or_path)
            processor = _CLIPProcessor.from_pretrained(model_name_or_path)
        else:
            raise ValueError(f"Invalid model_name_or_path {model_name_or_path}. Not supported by `clip_score` metric.")
        return model, processor

    raise ModuleNotFoundError(
        "`clip_score` metric requires `transformers` package be installed."
        " Either install with `pip install transformers>=4.10.0` or `pip install torchmetrics[multimodal]`."
    )

def _detect_modality(input_data: Union[Tensor, List[Tensor], List[str], str]) -> Literal["image", "text"]:
    """Automatically detect the modality of the input data.

    Args:
        input_data: Input data that can be either image tensors or text strings

    Returns:
        str: Either "image" or "text"

    Raises:
        ValueError: If the input_data is an empty list or modality cannot be determined

    """
    if isinstance(input_data, Tensor):
        return "image"

    if isinstance(input_data, list):
        if len(input_data) == 0:
            raise ValueError("Empty input list")
        if isinstance(input_data[0], Tensor):
            return "image"
        if isinstance(input_data[0], str):
            return "text"

    if isinstance(input_data, str):
        return "text"

    raise ValueError("Could not automatically determine modality for input_data")

def _process_image_data(images: Union[Tensor, List[Tensor]]) -> List[Tensor]:
    """Helper function to process image data."""
    images = [images] if not isinstance(images, list) and images.ndim == 3 else list(images)
    if not all(i.ndim == 3 for i in images):
        raise ValueError("Expected all images to be 3d but found image that has either more or less")
    return images

def _process_text_data(texts: Union[str, List[str]]) -> List[str]:
    """Helper function to process text data."""
    if not isinstance(texts, list):
        texts = [texts]
    return texts

def _get_features(
    data: List[Union[Tensor, str]],
    modality: str,
    device: torch.device,
    model: "_CLIPModel",
    processor: "_CLIPProcessor",
) -> Tensor:
    """Get features from the CLIP model for either images or text.

    Args:
       data: List of input data (images or text)
       modality: String indicating the type of input data (must be either "image" or "text")
       device: Device to run the model on
       model: CLIP model instance
       processor: CLIP processor instance

    Returns:
       Tensor of features from the CLIP model

    Raises:
        ValueError: If modality is not "image" or "text"

    """
    if modality == "image":
        image_data = [i for i in data if isinstance(i, Tensor)]  # Add type checking for images
        processed = processor(images=[i.cpu() for i in image_data], return_tensors="pt", padding=True)
        return model.get_image_features(processed["pixel_values"].to(device))
    if modality == "text":
        processed = processor(text=data, return_tensors="pt", padding=True)
        if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "max_position_embeddings"):
            max_position_embeddings = model.config.text_config.max_position_embeddings
            if processed["attention_mask"].shape[-1] > max_position_embeddings:
                rank_zero_warn(
                    f"Encountered caption longer than {max_position_embeddings=}. Will truncate captions to this"
                    "length. If longer captions are needed, initialize argument `model_name_or_path` with a model that"
                    "supports longer sequences.",
                    UserWarning,
                )
                processed["attention_mask"] = processed["attention_mask"][..., :max_position_embeddings]
                processed["input_ids"] = processed["input_ids"][..., :max_position_embeddings]
        return model.get_text_features(processed["input_ids"].to(device), processed["attention_mask"].to(device))
    raise ValueError(f"invalid modality {modality}")

def _clip_score_update(
    source: Union[Tensor, List[Tensor], List[str], str],
    target: Union[Tensor, List[Tensor], List[str], str],
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> tuple[Tensor, int]:
    """Update function for CLIP Score."""
    source_modality = _detect_modality(source)
    target_modality = _detect_modality(target)

    source_data = (
        _process_image_data(cast(Union[Tensor, List[Tensor]], source))
        if source_modality == "image"
        else _process_text_data(cast(Union[str, List[str]], source))
    )
    target_data = (
        _process_image_data(cast(Union[Tensor, List[Tensor]], target))
        if target_modality == "image"
        else _process_text_data(cast(Union[str, List[str]], target))
    )

    if len(source_data) != len(target_data):
        raise ValueError(
            "Expected the number of source and target examples to be the same but got "
            f"{len(source_data)} and {len(target_data)}"
        )

    device = (
        source_data[0].device
        if source_modality == "image" and isinstance(source_data[0], Tensor)
        else target_data[0].device
        if target_modality == "image" and isinstance(target_data[0], Tensor)
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)

    source_features = _get_features(
        cast(List[Union[Tensor, str]], source_data), source_modality, device, model, processor
    )
    target_features = _get_features(
        cast(List[Union[Tensor, str]], target_data), target_modality, device, model, processor
    )
    source_features = source_features / source_features.norm(p=2, dim=-1, keepdim=True)
    target_features = target_features / target_features.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity
    score = 100 * (source_features * target_features).sum(axis=-1)
    score = score.cpu() if source_modality == "text" and target_modality == "text" else score
    return score, len(source_data)