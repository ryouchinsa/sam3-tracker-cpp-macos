import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import time

class Sam3TrackerONNXInference:
    def __init__(
        self,
        vision_encoder_path: str,
        mask_decoder_path: str,
        device: str = "cuda",
    ):
        self.image_height = 1008
        self.image_width = 1008
        self.mask_height = 288
        self.mask_width = 288

        self.image_embeddings = None
        self.high_res_features1 = None
        self.high_res_features2 = None

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        t = time.perf_counter()
        self.vision_encoder  = ort.InferenceSession(vision_encoder_path,  providers=providers)
        self.mask_decoder    = ort.InferenceSession(mask_decoder_path,    providers=providers)
        print(f"load time:        {(time.perf_counter() - t) * 1000:.2f} ms")

    def preprocess_image(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Resize and normalize image.
        image: HxWx3 uint8 RGB
        Returns: (1,3,H,W) float32 in [-1,1], original (H,W)
        """
        from PIL import Image as PILImage
        pil = PILImage.fromarray(image)
        resized = np.array(
            pil.resize((self.image_width, self.image_height), PILImage.BILINEAR)
        )
        normalized = resized.astype(np.float32) / 127.5 - 1.0
        tensor = normalized.transpose(2, 0, 1)[np.newaxis]   # (1,3,H,W)
        return tensor

    def scale_points(
        self,
        points: list,
        orig_w: int,
        orig_h: int,
    ) -> np.ndarray:
        """Scale pixel-space points to model input resolution."""
        sx = self.image_width  / orig_w
        sy = self.image_height / orig_h
        arr = np.array(points, dtype=np.float32)   # (N,2)
        arr[:, 0] *= sx
        arr[:, 1] *= sy
        return arr

    def prepare_input_mask(
        self,
        mask: Optional[np.ndarray],
        orig_w: int,
        orig_h: int,
    ) -> np.ndarray:
        h_mask, w_mask = self.mask_height, self.mask_width
        if mask is None:
            return np.zeros((1, 1, h_mask, w_mask), dtype=np.float32)
        if mask.dtype == bool or mask.dtype == np.bool_:
            logits = np.where(mask.astype(np.float32) > 0, 20.0, -20.0).astype(np.float32)
        else:
            logits = mask.astype(np.float32)
        resized = cv2.resize(logits, (w_mask, h_mask), interpolation=cv2.INTER_LINEAR)
        return resized[np.newaxis, np.newaxis]   # (1, 1, H_mask, W_mask)

    def decode_masks(
        self,
        points: np.ndarray,
        point_labels: np.ndarray,
        input_masks: np.ndarray,
        has_mask_input: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_points = points.reshape(1, 1, -1, 2).astype(np.float32)
        input_labels = point_labels.reshape(1, 1, -1).astype(np.int64)
        print(input_points.shape)
        print(input_labels.shape)
        pred_masks, iou_scores, object_score_logits = self.mask_decoder.run(
            None,
            {
                "image_embeddings": self.image_embeddings,
                "high_res_features1": self.high_res_features1,
                "high_res_features2": self.high_res_features2,
                "input_points": input_points,
                "input_labels": input_labels,
                "input_masks":  input_masks,
                "has_mask_input":  has_mask_input,
            },
        )
        return pred_masks, iou_scores, object_score_logits

    def predict(
        self,
        image:        np.ndarray,
        points:       list,
        point_labels: list,
        input_mask:   Optional[np.ndarray] = None,  # (H,W) bool or float — prior mask
    ) -> dict:
        orig_h, orig_w = image.shape[:2]

        if self.image_embeddings is None:
            t = time.perf_counter()
            pixel_values = self.preprocess_image(image)
            print(f"preprocess_image: {(time.perf_counter()-t)*1000:.2f} ms")

            t = time.perf_counter()
            self.image_embeddings, self.high_res_features1, self.high_res_features2 = self.vision_encoder.run(None, {"input": pixel_values})
            print(f"encode_image:     {(time.perf_counter()-t)*1000:.2f} ms")
            print(self.image_embeddings.shape)
            print(self.high_res_features1.shape)
            print(self.high_res_features2.shape)
        
        pts_scaled = self.scale_points(points, orig_w, orig_h)
        pt_labels_arr = np.array(point_labels, dtype=np.int64)

        input_masks_tensor = self.prepare_input_mask(
            input_mask, orig_w, orig_h
        )
        if input_mask is None:
            has_mask_input = np.array([0], dtype=np.float32)
        else:
            has_mask_input = np.array([1], dtype=np.float32)

        t = time.perf_counter()
        pred_masks, iou_scores, object_score_logits = self.decode_masks(
            pts_scaled, pt_labels_arr, input_masks_tensor, has_mask_input
        )
        print(f"decode_masks:     {(time.perf_counter()-t)*1000:.2f} ms")

        print(pred_masks.shape)
        print(iou_scores.shape)
        print(object_score_logits.shape)

        t = time.perf_counter()
        result = self.postprocess(
            pred_masks, iou_scores, object_score_logits, (orig_h, orig_w)
        )
        print(f"postprocess:      {(time.perf_counter()-t)*1000:.2f} ms")
        return result

    def postprocess(
        self,
        pred_masks: np.ndarray,
        iou_scores: np.ndarray,
        object_score_logits: np.ndarray,
        orig_size: tuple[int, int],
    ) -> dict:
        orig_h, orig_w = orig_size
        pred_mask    = pred_masks[0][0][0]
        low_res_mask = pred_mask
        pred_mask    = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        pred_mask    = pred_mask > 0.0
        iou_score    = iou_scores[0][0][0]
        object_logit = object_score_logits[0][0][0]
        object_logit = 1.0 / (1.0 + np.exp(-object_logit))  # sigmoid
        return {
            "pred_mask": pred_mask,
            "low_res_mask": low_res_mask,
            "iou_score": iou_score,
            "object_logit": object_logit,
            "orig_size": orig_size,
        }

def visualize_results(
    image: np.ndarray,
    points: list, 
    point_labels: list,
    result: dict,
    output_path: str,
    alpha: float = 0.35,
):
    vis = image.copy()
    colors = [
        (30, 144, 255),
        (255, 144, 30),
        (144, 255, 30),
        (255, 30, 144),
        (30, 255, 144),
        (144, 30, 255),
        (255, 255, 30),
    ]
    mask = result["pred_mask"]
    iou = result["iou_score"]
    obj = result["object_logit"]
    print(iou, obj)
    color = colors[0]
    color_negative = colors[1]
    overlay = vis.copy()
    overlay[mask] = color
    vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(vis, contours, -1, color, 2)
    # Find top-left of mask for label placement
    ys, xs = np.where(mask)
    if len(xs):
        tx, ty = int(xs.min()), max(int(ys.min()) - 5, 10)
        cv2.putText(
            vis,
            f"iou:{iou:.2f} obj:{obj:.2f}",
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    # Draw input point(s) for this object if provided
    for index, pt in enumerate(points):
        label = point_labels[index]
        cx, cy = int(pt[0]), int(pt[1])
        cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
        if label == 1:
            cv2.circle(vis, (cx, cy), 6, color, 2)
        else:
            cv2.circle(vis, (cx, cy), 6, color_negative, 2)
    cv2.imwrite(output_path, vis)
    print(f"Saved: {output_path}")

def parse_point_prompts(point_str: str) -> tuple[list, list]:
    """
    Format: pos:x,y;neg:x,y;x,y
    Returns points (N,2) in pixel coords and labels (N,) as 1/0.
    """
    points, labels = [], []
    for part in point_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if part.startswith("pos:"):
            label, coords = 1, part[4:]
        elif part.startswith("neg:"):
            label, coords = 0, part[4:]
        else:
            label, coords = 1, part
        x, y = [float(v) for v in coords.split(",")]
        points.append([x, y])
        labels.append(label)
    return points, labels

def main():
    parser = argparse.ArgumentParser(description="Sam3Tracker ONNX Inference")
    parser.add_argument("--image",  type=str, required=True)
    parser.add_argument(
        "--points", type=str,
        help="Point prompts: pos:x,y;neg:x,y  (pixel coords)"
    )
    parser.add_argument(
        "--points_second", type=str,
        help="Point prompts: pos:x,y;neg:x,y  (pixel coords)"
    )
    parser.add_argument(
        "--points_third", type=str,
        help="Point prompts: pos:x,y;neg:x,y  (pixel coords)"
    )
    parser.add_argument(
        "--input-mask", type=str, default=None,
        help="Path to a prior mask image (grayscale PNG/JPG). White pixels = "
             "foreground. Resized automatically to mask_input_size. "
             "Use this to pass a mask from the previous frame when tracking."
    )
    parser.add_argument(
        "--model-dir", type=str, default="onnx-models-tracker",
        help="Directory containing vision-encoder.onnx, prompt-encoder.onnx, mask-decoder.onnx"
    )
    parser.add_argument("--output", type=str,   default="output-tracker.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.points:
        parser.error("Please specify --points")

    model_dir = Path(args.model_dir)
    engine = Sam3TrackerONNXInference(
        vision_encoder_path=str(model_dir / "vision-encoder.onnx"),
        mask_decoder_path=str(model_dir / "mask-decoder.onnx"),
        device=args.device
    )

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"Cannot load image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    points, point_labels = parse_point_prompts(args.points)

    # Load optional prior mask from disk (e.g. output of a previous frame).
    input_mask = None
    if args.input_mask:
        mask_bgr = cv2.imread(args.input_mask, cv2.IMREAD_GRAYSCALE)
        if mask_bgr is None:
            raise ValueError(f"Cannot load mask: {args.input_mask}")
        input_mask = mask_bgr > 127   # (H, W) bool

    t = time.perf_counter()
    result = engine.predict(
        image_rgb,
        points=points,
        point_labels=point_labels,
        input_mask=input_mask,
    )
    print(f"predict:          {(time.perf_counter()-t)*1000:.2f} ms")

    points_second = None
    if args.points_second:
        t = time.perf_counter()
        points_second, point_labels_second = parse_point_prompts(args.points_second)
        result = engine.predict(
            image_rgb,
            points=points_second,
            point_labels=point_labels_second,
            input_mask=result["low_res_mask"],
        )
        print(f"predict:          {(time.perf_counter()-t)*1000:.2f} ms")

    points_third = None
    if args.points_third:
        t = time.perf_counter()
        points_third, point_labels_third = parse_point_prompts(args.points_third)
        result = engine.predict(
            image_rgb,
            points=points_third,
            point_labels=point_labels_third,
            input_mask=result["low_res_mask"],
        )
        print(f"predict:          {(time.perf_counter()-t)*1000:.2f} ms")

    if points_second:
        points = points + points_second
        point_labels = point_labels + point_labels_second
    if points_third:
        points = points + points_third
        point_labels = point_labels + point_labels_third
    visualize_results(image_bgr, points, point_labels, result, args.output)

    mask_out = (result["pred_mask"].astype(np.uint8) * 255)
    mask_save_path = Path(args.output).stem + "_mask.png"
    cv2.imwrite(mask_save_path, mask_out)
    print(f"Saved prior mask for next frame: {mask_save_path}")
        

if __name__ == "__main__":
    main()
