"""
vlm8b 数据管线 — SFT jsonl → (pixel_values, input_ids, labels)
==============================================================

数据源（remote）: /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/*.jsonl
字段：`image`（相对根目录）、`messages_short`、`gt`（评测 GT 类名）。

文本布局（视觉前缀由 `model.expand_budget_rows()` 拼）::

    [system][user][assistant header]  = prompt（labels 全 -100）
    [answer][eos]                     = answer（labels 保留 id）

图像变换：train = RRC(448,0.35–1.0)+HFlip；val = Resize(512)+CenterCrop(448)；
两者都 ImageNet Normalize。collator 只 right-pad B 张图；V 行展开在 model 内。
"""
import json
import os
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IGNORE_INDEX = -100
IMAGE_SIZE = 448
VAL_RESIZE = 512
MAX_TEXT_LENGTH = 1024          # messages_short 实测 ≤ ~710
MESSAGES_KEY = "messages_short"
IMAGENET_MEAN = (0.485, 0.456, 0.406)      # 与 model_v2_linattn 同值（[0,1] 标度）
IMAGENET_STD = (0.229, 0.224, 0.225)


# ═══════════════════════════════════════════════════════════════
# 变换
# ═══════════════════════════════════════════════════════════════

def build_image_transform(train: bool):
    """训练 RRC(448, 0.35–1.0)+HFlip；验证 Resize(512)+CenterCrop(448)；均 Normalize。"""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                IMAGE_SIZE, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3),
                interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(VAL_RESIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class SelfDistillSFTDataset(Dataset):
    """`sft/*.jsonl` → `(pixel_values, input_ids, attention_mask, labels, gt_name)`。"""

    def __init__(self, jsonl_path: str, tokenizer, image_root: str, transform):
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.transform = transform
        self.eos_token_id = tokenizer.eos_token_id
        self.records: List[Dict[str, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    self.records.append({
                        "image": row["image"],
                        "messages": row[MESSAGES_KEY],
                        "gt": str(row.get("gt", "")),
                    })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        with Image.open(os.path.join(
                self.image_root, record["image"])) as image:
            pixel_values = self.transform(image.convert("RGB"))

        messages = json.loads(record["messages"]) if isinstance(
            record["messages"], str) else record["messages"]
        prompt_ids = self._chat_ids(messages[:-1], add_generation_prompt=True)
        full_ids = self._chat_ids(messages)
        if full_ids[:len(prompt_ids)] == prompt_ids:
            answer_ids = full_ids[len(prompt_ids):]
        else:                                   # chat template 前缀不一致时兜底
            answer_ids = list(self.tokenizer(
                messages[-1]["content"], add_special_tokens=False)["input_ids"])
        answer_ids = self._truncate_at_eos(answer_ids)

        ids = prompt_ids + answer_ids
        assert len(ids) <= MAX_TEXT_LENGTH, f"文本 {len(ids)} > {MAX_TEXT_LENGTH}"
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor(
                [IGNORE_INDEX] * len(prompt_ids) + answer_ids, dtype=torch.long),
            "gt_name": record["gt"] or os.path.dirname(record["image"]),
        }

    def _chat_ids(self, messages, add_generation_prompt: bool = False) -> List[int]:
        """`apply_chat_template(tokenize=True)` → id 列表。

        transformers 5.x 在 tokenize=True 时返回 BatchEncoding（4.x 返回 list[int]），
        `list(...)` 会拿到键名字符串。这里统一取出 input_ids。
        """
        out = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt)
        if hasattr(out, "keys") and "input_ids" in out:
            out = out["input_ids"]
        return [int(token) for token in out]

    def _truncate_at_eos(self, answer_ids: List[int]) -> List[int]:
        """answer 截到第一个 eos（含），丢掉模板尾部换行；没有 eos 则补一个。"""
        eos = self.eos_token_id
        if eos is None:
            return list(answer_ids)
        if eos in answer_ids:
            return list(answer_ids[:answer_ids.index(eos) + 1])
        return list(answer_ids) + [eos]


# ═══════════════════════════════════════════════════════════════
# Collator —— 只做 B 张图的 right-padding（V 行展开在 model 内）
# ═══════════════════════════════════════════════════════════════

class EightBudgetCollator:
    """B 个样本 right-pad 成批；batch 维度是**图片数 B**，不是 B×V。"""

    def __init__(self, tokenizer):
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, object]:
        max_len = max(int(feature["input_ids"].numel()) for feature in features)
        batch = {
            "input_ids": torch.full(
                (len(features), max_len), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((len(features), max_len), dtype=torch.long),
            "labels": torch.full(
                (len(features), max_len), IGNORE_INDEX, dtype=torch.long),
        }
        for row, feature in enumerate(features):
            length = int(feature["input_ids"].numel())
            for key in ("input_ids", "attention_mask", "labels"):
                batch[key][row, :length] = feature[key]
        batch["pixel_values"] = torch.stack(
            [feature["pixel_values"] for feature in features], dim=0)
        batch["gt_name"] = [str(feature.get("gt_name", "")) for feature in features]
        return batch
