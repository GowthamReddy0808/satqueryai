"""
SatQuery AI - Teammate Model Adapter & Integration Interface
Provides a standardized client for the specialized fine-tuned Remote Sensing Vision-Language Model.
Supports custom endpoints, Bearer auth, health-checks, and honest disconnected states.
"""

import os
import time
import base64
from typing import Dict, Any, List, Optional
import requests
from core.schema import BoundingBox, TaskType


class TeammateModelAdapter:
    """
    Standardized adapter for integrating the teammate's specialized remote sensing model.
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_id: str = "isro-satquery-rs-v1",
        timeout_sec: int = 45
    ):
        self.api_base_url = (
            api_base_url or
            os.getenv("TEAMMATE_MODEL_API_URL") or
            "http://localhost:8000"
        ).rstrip("/")
        self.api_key = api_key or os.getenv("TEAMMATE_MODEL_API_KEY")
        self.model_id = model_id
        self.timeout_sec = timeout_sec

    def check_health(self) -> Dict[str, Any]:
        """
        Ping teammate API health check endpoint to determine availability.
        """
        start = time.time()
        url = f"{self.api_base_url}/health"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            res = requests.get(url, headers=headers, timeout=3)
            elapsed_ms = round((time.time() - start) * 1000, 1)
            if res.ok:
                return {
                    "available": True,
                    "status_code": res.status_code,
                    "latency_ms": elapsed_ms,
                    "model_id": self.model_id,
                    "details": res.json() if res.headers.get("content-type", "").startswith("application/json") else "OK"
                }
            return {
                "available": False,
                "status_code": res.status_code,
                "latency_ms": elapsed_ms,
                "error": f"Endpoint returned HTTP {res.status_code}"
            }
        except requests.exceptions.ConnectionError:
            return {
                "available": False,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "error": f"Connection refused at {self.api_base_url}. Inference service is offline."
            }
        except Exception as e:
            return {
                "available": False,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "error": str(e)
            }

    def infer(
        self,
        task_type: TaskType,
        query: str,
        image_bytes_list: List[bytes],
        image_formats: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute inference against teammate's specialized model API endpoint.
        """
        health = self.check_health()
        if not health["available"]:
            return {
                "success": False,
                "error": f"Teammate RS Model is unavailable at {self.api_base_url}: {health.get('error')}",
                "is_unavailable": True
            }

        # Encode images to base64
        images_payload = []
        for idx, (b, fmt) in enumerate(zip(image_bytes_list, image_formats)):
            b64_str = base64.b64encode(b).decode("utf-8")
            images_payload.append({
                "index": idx,
                "format": fmt,
                "data_base64": b64_str
            })

        payload = {
            "model": self.model_id,
            "task_type": task_type.value,
            "query": query,
            "images": images_payload,
            "metadata": metadata or {}
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start_time = time.time()
        try:
            url = f"{self.api_base_url}/v1/infer"
            res = requests.post(url, json=payload, headers=headers, timeout=self.timeout_sec)
            elapsed_sec = round(time.time() - start_time, 2)

            if not res.ok:
                return {
                    "success": False,
                    "error": f"Teammate model API error ({res.status_code}): {res.text[:400]}",
                    "duration": elapsed_sec
                }

            data = res.json()
            # Parse detections if returned
            detections = []
            for d in data.get("detections", []):
                detections.append(BoundingBox(
                    box_2d=d.get("box_2d", []),
                    label=d.get("label", "detected_feature"),
                    confidence=d.get("confidence"),
                    color=d.get("color", "#38bdf8")
                ))

            return {
                "success": True,
                "answer": data.get("answer", ""),
                "reasoning": data.get("reasoning"),
                "detections": detections,
                "confidence": data.get("confidence"),
                "duration": elapsed_sec,
                "execution_trace": data.get("execution_trace", [])
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Teammate API communication error: {e}",
                "duration": round(time.time() - start_time, 2)
            }
