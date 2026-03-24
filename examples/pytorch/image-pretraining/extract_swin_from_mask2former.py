#!/usr/bin/env python3
"""extract_swin_from_mask2former.py

Extract the Swin backbone from a Mask2Former checkpoint and save it as a
standalone SwinForMaskedImageModeling model that can be used directly as
--model_name_or_path in run_mim.py.

Usage
-----
python extract_swin_from_mask2former.py \
    --mask2former_name_or_path facebook/mask2former-swin-base-ade-semantic \
    --out_dir swin_base_from_mask2former_ade \
    [--encoder_stride 32]
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract(mask2former_name_or_path: str, out_dir: str, encoder_stride: int):
    # Lazy imports so the script fails early with a clear message if transformers
    # is not installed.
    from transformers import (
        Mask2FormerForUniversalSegmentation,
        SwinForMaskedImageModeling,
    )

    # 1. Load full Mask2Former model
    logger.info(f"Loading Mask2Former from: {mask2former_name_or_path}")
    m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
        mask2former_name_or_path
    )
    m2f.eval()

    # 2. Locate the Swin backbone inside Mask2Former.
    #    Older transformers versions wrap it in a SwinModel at encoder.model;
    #    newer versions use SwinBackbone directly at encoder (no .model child).
    encoder = m2f.model.pixel_level_module.encoder
    if hasattr(encoder, "model"):
        # Old API: encoder is a wrapper, actual SwinModel at encoder.model
        swin_model = encoder.model
        logger.info("Detected old-style Mask2Former (encoder.model = SwinModel)")
    else:
        # New API: encoder IS the SwinBackbone
        swin_model = encoder
        logger.info("Detected new-style Mask2Former (encoder = SwinBackbone)")

    logger.info(
        f"Found Swin backbone: hidden_size={swin_model.config.hidden_size}, "
        f"depths={swin_model.config.depths}, "
        f"num_heads={swin_model.config.num_heads}"
    )

    # 3. Build a SwinForMaskedImageModeling with the same config.
    #    Only the MIM decoder head (a small pixel-shuffle conv) is randomly
    #    initialised; the encoder weights come from the Mask2Former backbone.
    from transformers import SwinConfig
    # SwinBackbone uses a different config class; convert to plain SwinConfig
    swin_cfg_dict = swin_model.config.to_dict()
    swin_cfg_dict.pop("model_type", None)
    swin_config = SwinConfig(**{k: v for k, v in swin_cfg_dict.items()
                                if k in SwinConfig().to_dict()})
    swin_config.encoder_stride = encoder_stride   # required by SwinForMaskedImageModeling

    mim_model = SwinForMaskedImageModeling(swin_config)

    # 4. Copy backbone weights.
    #    SwinBackbone and SwinModel share the same sub-modules
    #    (embeddings / encoder / layernorm), so their state-dict keys match.
    backbone_sd = swin_model.state_dict()
    # Filter out pooler if present (SwinModel has it, SwinBackbone may not)
    missing, unexpected = mim_model.swin.load_state_dict(
        backbone_sd, strict=False
    )
    if missing:
        logger.warning(f"Missing keys when loading backbone: {missing[:10]}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading backbone: {unexpected[:10]}")
    logger.info("Backbone weights loaded successfully.")

    # 5. Save as a standard HF checkpoint
    mim_model.save_pretrained(out_dir)
    logger.info(f"Saved SwinForMaskedImageModeling to: {out_dir}")
    logger.info(
        "\nYou can now use it in run_mim.py with:\n"
        f"  --model_name_or_path {out_dir} \\\n"
        f"  --patch_size {swin_config.patch_size} \\\n"
        f"  --encoder_stride {encoder_stride} \\\n"
        f"  --mask_patch_size 32"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract Swin backbone from a Mask2Former checkpoint for MIM pretraining."
    )
    parser.add_argument(
        "--mask2former_name_or_path",
        type=str,
        required=True,
        help="HF hub name or local path of the Mask2Former model.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save the extracted SwinForMaskedImageModeling model.",
    )
    parser.add_argument(
        "--encoder_stride",
        type=int,
        default=32,
        help=(
            "Total spatial stride of the Swin encoder "
            "(patch_size × 2^(num_stages-1)). "
            "Default 32 is correct for Swin-T/S/B/L with patch_size=4 and 4 stages."
        ),
    )
    args = parser.parse_args()
    extract(args.mask2former_name_or_path, args.out_dir, args.encoder_stride)


if __name__ == "__main__":
    main()
