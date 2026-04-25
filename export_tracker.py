import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.sam3_tracker.modeling_sam3_tracker import Sam3TrackerModel
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

class TrackerVisionEncoderWrapper(nn.Module):
    def __init__(self, model: Sam3TrackerModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        image_embeddings = self.model.get_image_embeddings(pixel_values)
        high_res_features1 = image_embeddings[0]
        high_res_features2 = image_embeddings[1]
        image_embeddings = image_embeddings[2]
        return image_embeddings, high_res_features1, high_res_features2

class TrackerMaskDecoderWrapper(nn.Module):
    def __init__(self, model: Sam3TrackerModel):
        super().__init__()
        self.model = model
        self.prompt_encoder = model.prompt_encoder
        self.mask_decoder = model.mask_decoder

    def forward(
        self,
        image_embeddings: torch.Tensor,   # (HW2, N, C)     low-res
        high_res_features1: torch.Tensor, # (HW0, N, C//8)  high-res 0
        high_res_features2: torch.Tensor, # (HW1, N, C//4)  high-res 1
        input_points: torch.Tensor,       # (B, P, N, 2)
        input_labels: torch.Tensor,       # (B, P, N)       int64
        input_masks:  torch.Tensor,       # (B, 1, H, W)    float32 — prior mask logits
        has_mask_input: torch.Tensor,     # [1]
    ):
        batch_size = input_points.shape[0]
        sparse_embeddings = self.prompt_encoder._embed_points(input_points, input_labels, pad=True)
        dense_embeddings = has_mask_input * self.prompt_encoder.mask_embed(input_masks)
        dense_embeddings = dense_embeddings + (1 - has_mask_input) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size, -1, self.prompt_encoder.image_embedding_size[0], self.prompt_encoder.image_embedding_size[1]
        )
        image_positional_embeddings = self.model.get_image_wide_positional_embeddings()
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
        low_res_masks, iou_scores, _, object_score_logits = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            high_resolution_features=[high_res_features1, high_res_features2],
            attention_similarity=None,
            target_embedding=None,
        )
        # low_res_masks:        (B, P, num_masks, H, W)
        # iou_scores:           (B, P, num_masks)
        # object_score_logits:  (B, P, 1)
        return low_res_masks, iou_scores, object_score_logits

def export_vision_encoder(
    model: Sam3TrackerModel,
    output_dir: Path,
    device: str = "cuda",
    quantize: bool = False,
):
    wrapper = TrackerVisionEncoderWrapper(model).to(device).eval()
    dummy_pixel = torch.randn(1, 3, 1008, 1008, device=device)
    with torch.no_grad():
        image_embeddings, high_res_features1, high_res_features2 = wrapper(dummy_pixel)
    print(f"  image_embeddings   shape: {image_embeddings.shape}")
    print(f"  high_res_features1 shape: {high_res_features1.shape}")
    print(f"  high_res_features2 shape: {high_res_features2.shape}")
    torch.onnx.export(
        wrapper,
        (dummy_pixel,),
        str(output_dir / "vision-encoder.onnx"),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        input_names=["input"],
        output_names=["image_embeddings", "high_res_features1", "high_res_features2"],
    )
    print(f"  Exported vision encoder -> {output_dir / 'vision-encoder.onnx'}")
    if quantize:
        quantize_dynamic(
            model_input=str(output_dir / "vision-encoder.onnx"),
            model_output=str(output_dir / "vision-encoder.onnx"),
            op_types_to_quantize=["MatMul"],
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("  Quantized vision encoder.")

def export_mask_decoder(
    model: Sam3TrackerModel,
    output_dir: Path,
    device: str = "cuda",
    quantize: bool = False,
):
    wrapper = TrackerMaskDecoderWrapper(model).to(device).eval()
    feat_sizes = model.backbone_feature_sizes
    h0, w0 = feat_sizes[0]
    h1, w1 = feat_sizes[1]
    h2, w2 = feat_sizes[2]
    hidden = model.hidden_dim
    c_s0 = hidden // 8
    c_s1 = hidden // 4
    dummy_feat_s0 = torch.randn(1, c_s0, h0, w0, device=device)
    dummy_feat_s1 = torch.randn(1, c_s1, h1, w1, device=device)
    dummy_feat_s2 = torch.randn(1, hidden, h2, w2, device=device)
    dummy_points = torch.rand(1, 1, 1, 2, device=device)
    dummy_labels = torch.ones(1, 1, 1, dtype=torch.int64, device=device)
    mask_h, mask_w = model.prompt_encoder.mask_input_size
    dummy_masks = torch.zeros(1, 1, mask_h, mask_w, device=device)
    has_mask_input = torch.tensor([1], dtype=torch.float)
    with torch.no_grad():
        low_res_masks, iou_scores, object_score_logits = wrapper(dummy_feat_s2, dummy_feat_s0, dummy_feat_s1, dummy_points, dummy_labels, dummy_masks, has_mask_input)
    print(f"  low_res_masks          shape: {low_res_masks.shape}")
    print(f"  iou_scores             shape: {iou_scores.shape}")
    print(f"  object_score_logits    shape: {object_score_logits.shape}")
    torch.onnx.export(
        wrapper,
        (dummy_feat_s2, dummy_feat_s0, dummy_feat_s1, dummy_points, dummy_labels, dummy_masks, has_mask_input),
        str(output_dir / "mask-decoder.onnx"),
        input_names=[
            "image_embeddings", "high_res_features1", "high_res_features2", "input_points", "input_labels", "input_masks", "has_mask_input"
        ],
        output_names=["low_res_masks", "iou_scores", "object_score_logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input_points":{2: "num_points"},
            "input_labels":{2: "num_points"}
        },
    )
    print(f"  Exported mask decoder -> {output_dir / 'mask-decoder.onnx'}")
    if quantize:
        quantize_dynamic(
            model_input=str(output_dir / "mask-decoder.onnx"),
            model_output=str(output_dir / "mask-decoder.onnx"),
            op_types_to_quantize=["MatMul"],
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("  Quantized mask decoder.")

def main():
    parser = argparse.ArgumentParser(description="Export Sam3TrackerModel to ONNX")
    parser.add_argument(
        "--module", type=str, choices=["vision", "decoder"], default=None,
    )
    parser.add_argument("--all", action="store_true", help="Export all modules")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="onnx-models-tracker")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    if not args.module and not args.all:
        parser.error("Please specify --module or --all")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path} ...")
    model = (
        Sam3TrackerModel.from_pretrained(args.model_path)
        .to(args.device)
        .eval()
    )
    print(f"  backbone_feature_sizes: {model.backbone_feature_sizes}")
    print(f"  hidden_dim: {model.hidden_dim}")
    print(f"  mask_input_size: {model.prompt_encoder.mask_input_size}")

    modules = ["vision", "decoder"] if args.all else [args.module]

    with torch.no_grad():
        for m in modules:
            if m == "vision":
                export_vision_encoder(
                    model, output_dir, args.device, args.quantize,
                )
            elif m == "decoder":
                export_mask_decoder(
                    model, output_dir, args.device, args.quantize,
                )


if __name__ == "__main__":
    main()
