"""
Comprehensive Integration & Verification Test Suite for SatQuery AI
Tests all core modules: raster processing, change detection, optical-SAR fusion,
annotation renderer, teammate model adapter, agentic orchestrator, and evaluation metrics.
"""

import os
import io
import unittest
import numpy as np
from PIL import Image

# Import SatQuery AI packages
from core.schema import TaskType, FeatureEvidenceState, BoundingBox, SatQueryResult
from core.orchestrator import AgenticQueryOrchestrator
from tools.raster_processor import inspect_raster_metadata, calculate_ndvi, generate_spectral_masks
from tools.change_detector import validate_bi_temporal_pair, compute_bi_temporal_change_map
from tools.optical_sar_fusion import process_optical_sar_pair
from tools.annotator import draw_bounding_boxes, create_segmentation_overlay
from models.teammate_adapter import TeammateModelAdapter
from evaluation.evaluator import compute_bbox_iou, compute_change_detection_metrics, compute_vqa_accuracy
from evaluation.benchmarks import BENCHMARK_DATASETS


class TestSatQueryAI(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 128x128 test scene (CIR: Red=NIR, Green=visible green, Blue=visible blue)
        arr = np.zeros((128, 128, 3), dtype=np.uint8)
        # Vegetation quadrant (high NIR/Red)
        arr[:64, :64] = [200, 40, 40]
        # Water body quadrant (low reflectance, dark)
        arr[:64, 64:] = [20, 30, 45]
        # Soil / sediment quadrant (balanced albedo)
        arr[64:, :64] = [150, 140, 130]
        # Built-up quadrant (bright cyan/grey)
        arr[64:, 64:] = [120, 180, 200]

        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.sample_bytes = buf.getvalue()
        self.sample_img = img

    def test_raster_metadata_inspection(self):
        meta = inspect_raster_metadata(self.sample_bytes, "test_scene.png")
        self.assertEqual(meta["width"], 128)
        self.assertEqual(meta["height"], 128)
        self.assertEqual(meta["band_count"], 3)
        self.assertIn("PNG", meta["format"])
        self.assertGreater(meta["size_kb"], 0)

    def test_geotiff_crs_and_resolution(self):
        """Test GeoTIFF CRS, bounds, and resolution parsing with rasterio."""
        import rasterio
        from rasterio.transform import from_origin

        transform = from_origin(700000, 2000000, 10.0, 10.0)
        profile = {
            "driver": "GTiff",
            "height": 64,
            "width": 64,
            "count": 4,
            "dtype": "uint8",
            "crs": "EPSG:32643",
            "transform": transform
        }
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                data = np.zeros((4, 64, 64), dtype=np.uint8)
                dst.write(data)
            tif_bytes = memfile.read()

        meta = inspect_raster_metadata(tif_bytes, "sentinel2_l2a.tif")
        self.assertTrue(meta["has_georeference"])
        self.assertIn("32643", meta["crs"])
        self.assertEqual(meta["band_count"], 4)
        self.assertIn("10.0m", meta["resolution"])
        self.assertIsNotNone(meta["bounds"])

    def test_spectral_masks_generation(self):
        masks = generate_spectral_masks(self.sample_bytes)
        self.assertIn("vegetation", masks)
        self.assertIn("water", masks)
        self.assertIn("soil", masks)
        self.assertIn("heatmap", masks)

        v_img, v_cov, v_desc = masks["vegetation"]
        self.assertIsInstance(v_img, Image.Image)
        self.assertGreater(v_cov, 0.0)

        w_img, w_cov, w_desc = masks["water"]
        self.assertIsInstance(w_img, Image.Image)
        self.assertGreater(w_cov, 0.0)

    def test_ndvi_scientific_honesty(self):
        """Verify that genuine NDVI is NOT falsely claimed for 3-band RGB imagery."""
        arr_idx, vis_img, report = calculate_ndvi(self.sample_bytes)
        self.assertFalse(report["is_genuine_ndvi"], "Must not claim genuine physical NDVI on standard 3-band imagery")
        self.assertIn("Calibrated Proxy", report["status_message"])
        self.assertGreater(report["vegetation_coverage_pct"], 0.0)

    def test_bitemporal_change_detection(self):
        # Create T2 by altering one quadrant (clearing vegetation)
        arr_t2 = np.array(self.sample_img)
        arr_t2[:64, :64] = [80, 80, 80]  # Cleared vegetation
        img_t2 = Image.fromarray(arr_t2)
        buf = io.BytesIO()
        img_t2.save(buf, format="PNG")
        t2_bytes = buf.getvalue()

        # Validate pair
        valid, msg, meta = validate_bi_temporal_pair(self.sample_bytes, t2_bytes)
        self.assertTrue(valid)

        # Compute change map
        heat, overlay, metrics = compute_bi_temporal_change_map(self.sample_bytes, t2_bytes)
        self.assertIsInstance(heat, Image.Image)
        self.assertIsInstance(overlay, Image.Image)
        self.assertGreater(metrics["total_changed_pct"], 15.0)
        self.assertGreater(metrics["loss_reduction_pct"], 10.0)

    def test_optical_sar_joint_analysis(self):
        # Create a synthetic SAR backscatter image
        sar_arr = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        sar_img = Image.fromarray(sar_arr)
        buf = io.BytesIO()
        sar_img.save(buf, format="PNG")
        sar_bytes = buf.getvalue()

        opt_vis, sar_vis, composite, report = process_optical_sar_pair(
            self.sample_bytes, sar_bytes, "Optical", "SAR"
        )
        self.assertIsInstance(composite, Image.Image)
        self.assertIn("optical_contribution", report)
        self.assertIn("sar_contribution", report)

    def test_bounding_box_annotation(self):
        detections = [
            BoundingBox(box_2d=[0.1, 0.1, 0.5, 0.5], label="vegetation_cluster", confidence=0.88),
            BoundingBox(box_2d=[0.6, 0.6, 0.9, 0.9], label="settlement", confidence=0.92)
        ]
        annotated = draw_bounding_boxes(self.sample_img, detections)
        self.assertIsInstance(annotated, Image.Image)
        self.assertEqual(annotated.size, self.sample_img.size)

    def test_teammate_adapter_offline_resilience(self):
        """Adapter must gracefully handle offline teammate server without crashing."""
        adapter = TeammateModelAdapter(api_base_url="http://127.0.0.1:9999", timeout_sec=2)
        health = adapter.check_health()
        self.assertFalse(health["available"])
        self.assertIn("refused", health.get("error", "").lower())

        # Inference should cleanly report unavailable state
        res = adapter.infer(
            task_type=TaskType.VQA,
            query="test",
            image_bytes_list=[self.sample_bytes],
            image_formats=["image/png"]
        )
        self.assertFalse(res["success"])
        self.assertTrue(res.get("is_unavailable"))

    def test_orchestrator_classification_and_abstention(self):
        orchestrator = AgenticQueryOrchestrator()

        # Classification tests
        self.assertEqual(orchestrator.classify_task("Where is vegetation visible?"), TaskType.VQA)
        self.assertEqual(orchestrator.classify_task("Locate the water body with bounding box"), TaskType.GROUNDING)
        self.assertEqual(orchestrator.classify_task("Calculate NDVI vegetation index"), TaskType.SPECTRAL)
        self.assertEqual(orchestrator.classify_task("Describe the scene caption"), TaskType.CAPTIONING)

        # Honest Abstention: Change detection query with only 1 image
        res = orchestrator.execute_workflow(
            query="What changed between these two dates?",
            image_bytes_list=[self.sample_bytes],
            image_names=["single_scene.png"],
            image_types=["image/png"]
        )
        self.assertEqual(res.status, "abstention")
        self.assertIn("Abstention", res.answer)
        self.assertEqual(res.evidence_state, FeatureEvidenceState.INPUT_UNSUITABLE)

    def test_evaluation_metrics(self):
        # Bbox IoU
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [0.0, 0.0, 10.0, 10.0]
        self.assertAlmostEqual(compute_bbox_iou(box1, box2), 1.0)

        box3 = [5.0, 0.0, 15.0, 10.0]
        iou_val = compute_bbox_iou(box1, box3)
        self.assertGreater(iou_val, 0.3)
        self.assertLess(iou_val, 0.4)

        # Change detection metrics
        gt_mask = np.array([True, True, False, False])
        pred_mask = np.array([True, False, True, False])
        metrics = compute_change_detection_metrics(pred_mask, gt_mask)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1_score"], 0.5)

        # VQA accuracy
        preds = ["Water body", "Forest canopy"]
        refs = ["Water body", "Forest"]
        vqa_res = compute_vqa_accuracy(preds, refs)
        self.assertEqual(vqa_res["exact_match_accuracy"], 1.0)

        # Datasets verification
        self.assertIn("BigEarthNet", BENCHMARK_DATASETS)
        self.assertIn("VRSBench", BENCHMARK_DATASETS)
        self.assertIn("RSVQA", BENCHMARK_DATASETS)
        self.assertIn("CDVQA", BENCHMARK_DATASETS)


if __name__ == "__main__":
    unittest.main()
