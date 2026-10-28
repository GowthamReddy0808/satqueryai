"""
SatQuery AI - Model Registry & Adapter Dispatcher
Manages registered models, connection health, and capabilities mapping.
"""

from typing import Dict, Any, List
from core.schema import TaskType
from .teammate_adapter import TeammateModelAdapter


class ModelRegistry:
    """Registry maintaining metadata, capabilities, and availability of models."""

    def __init__(self):
        self.teammate_adapter = TeammateModelAdapter()

    def get_registered_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "teammate_model",
                "name": "ISRO SatQuery RS-V1 (Teammate Fine-Tuned Model)",
                "provider": "Custom RS Model API",
                "type": "Specialized Remote-Sensing VLM",
                "endpoint": self.teammate_adapter.api_base_url,
                "supported_tasks": [
                    TaskType.VQA.value,
                    TaskType.CAPTIONING.value,
                    TaskType.GROUNDING.value,
                    TaskType.COUNTING.value,
                    TaskType.LANDCOVER.value,
                    TaskType.CHANGE_VQA.value,
                    TaskType.OPTICAL_SAR.value
                ],
                "is_specialized": True
            },
            {
                "id": "nvidia/nemotron-3.5-super-vl:free",
                "name": "Nvidia Nemotron 3.5 Super VL (Multimodal)",
                "provider": "OpenRouter",
                "type": "General Multi-modal Vision-Language",
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "supported_tasks": [
                    TaskType.VQA.value,
                    TaskType.CAPTIONING.value,
                    TaskType.LANDCOVER.value,
                    TaskType.SPECTRAL.value
                ],
                "is_specialized": False
            },
            {
                "id": "google/gemma-4-26b-a4b-it:free",
                "name": "Google Gemma 4 26B (Vision Multimodal)",
                "provider": "OpenRouter",
                "type": "General Multi-modal Vision-Language",
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "supported_tasks": [
                    TaskType.VQA.value,
                    TaskType.CAPTIONING.value
                ],
                "is_specialized": False
            },
            {
                "id": "qwen/qwen3-vl-8b-instruct",
                "name": "Qwen 3 VL 8B Instruct",
                "provider": "OpenRouter",
                "type": "Vision-Language Grounding Model",
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "supported_tasks": [
                    TaskType.VQA.value,
                    TaskType.GROUNDING.value,
                    TaskType.CAPTIONING.value
                ],
                "is_specialized": False
            }
        ]

    def check_all_statuses(self) -> Dict[str, Any]:
        """Check availability of connected model services."""
        teammate_health = self.teammate_adapter.check_health()
        return {
            "teammate_model": teammate_health,
            "openrouter": {
                "available": True,
                "endpoint": "https://openrouter.ai/api/v1"
            }
        }
