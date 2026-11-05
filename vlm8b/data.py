"""
vlm8b 数据管线 — IN-100 自蒸馏 SFT (`messages_short`) → 像素 + 文本 ids/labels
=================================================================================

数据源（remote）::

    /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/
        train_nothink_all.jsonl / val_nothink_all.jsonl
        train_think_all.jsonl  / val_think_all.jsonl      （或 *_strict）

每行 JSON 字段（用到 3 个）::

    image          相对 /root/autodl-tmp/datasets/imagenet100/ 的图片路径
    messages_short JSON 数组 [system, user, assistant]（训练/评测统一用 short）
    gt             GT 类名（评测 exact-match 用）

文本布局（**视觉前缀由 model.py 在 forward 里拼接**，这里只产文本部分）::

    [system][user][assistant header]  = prompt（labels 全 -100）
    [answer][eos]                     = answer（labels 保留 id）

图像变换（与阶段2 编码器训练口径一致，见 `data_linattn.py`）::

    train: RandomResizedCrop(448, scale=(0.35,1.0), BICUBIC) + HFlip
           + ToTensor + ImageNet Normalize
    val:   Resize(512, BICUBIC) + CenterCrop(448) + ToTensor + Normalize

模块结构::

    1. 常量
    2. build_image_transform()
    3. SelfDistillSFTDataset
    4. EightBudgetCollator

注意：**collator 只做 B 张图**；同一张图的 8 行展开在
`model.expand_budget_rows()` 里做（见设计 §4.1 的"8 行批"裁决）。
"""

import json
import os
import sys
from typing import Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════

IGNORE_INDEX: int = -100

DEFAULT_IMAGE_SIZE: int = 448
DEFAULT_MAX_TEXT_LENGTH: int = 1024     # messages_short 实测 ≤ ~710（think p90）
DEFAULT_MESSAGES_KEY: str = "messages_short"

#: val 变换的 Resize 边长比例（与阶段2 训练脚本一致：224→256，448→512）。
VAL_RESIZE_RATIO: float = 256.0 / 224.0

#: 仓库 `model_v2_linattn.py` 里的 ImageNet 统计量（[0,1] 标度）。导入失败时用
#: 下面的同值兜底（本仓库一定带该文件，兜底只用于 vlm8b/ 被单独拷走的情形）。
_IMAGENET_MEAN_FALLBACK: tuple = (0.485, 0.456, 0.406)
_IMAGENET_STD_FALLBACK: tuple = (0.229, 0.224, 0.225)


# ═══════════════════════════════════════════════════════════════
# 变换
# ═══════════════════════════════════════════════════════════════

def _imagenet_stats() -> tuple:
    """返回 (mean, std)，优先用 `model_v2_linattn.IMAGENET_MEAN/STD`。"""
    try:
        from model_v2_linattn import IMAGENET_MEAN, IMAGENET_STD
        return tuple(IMAGENET_MEAN), tuple(IMAGENET_STD)
    except ImportError:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from model_v2_linattn import IMAGENET_MEAN, IMAGENET_STD
            return tuple(IMAGENET_MEAN), tuple(IMAGENET_STD)
        except ImportError:
            return _IMAGENET_MEAN_FALLBACK, _IMAGENET_STD_FALLBACK


def build_image_transform(
    train: bool,
    img_size: int = DEFAULT_IMAGE_SIZE,
    augment: bool = True,
):
    """构造训练 / 验证图像变换（与阶段2 编码器原训练一致）。

    Args:
        train: True = 训练口径；False = 验证口径（确定性）。
        img_size: 编码器输入边长（本实验 448）。
        augment: 仅在 `train=True` 时有意义；False ⇒ 用确定性 Resize+CenterCrop
            （设计 §12 签字点 4 的"不要增强"分支）。

    Returns:
        torchvision.transforms.Compose
    """
    mean, std = _imagenet_stats()
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3),
                interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(round(img_size * VAL_RESIZE_RATIO)),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class SelfDistillSFTDataset(Dataset):
    """IN-100 自蒸馏 SFT jsonl → (pixel_values, input_ids, labels, gt_name)。

    只在内存里保留 ``image / messages_short / gt`` 三个字段（丢掉体积大的
    `messages` 全量版），121k 行约占 ~100MB。

    Args:
        jsonl_path: `train_nothink_all.jsonl` 等。
        tokenizer: Qwen tokenizer（`apply_chat_template` 用）。
        image_root: 图片根目录（`row["image"]` 是相对路径）。
        transform: `build_image_transform()` 的输出。
        messages_key: 默认 `messages_short`（训练/评测统一口径）。
        max_text_length: 文本（prompt+answer）上限；超出直接报错而不是静默截断
            （实测 think p90 ≈ 705，1024 有充足余量）。
        append_eos: answer 若未以 eos 结束则补一个（让 greedy 生成会停）。
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        image_root: str,
        transform,
        messages_key: str = DEFAULT_MESSAGES_KEY,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
        append_eos: bool = True,
    ):
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"找不到 SFT jsonl: {jsonl_path}")
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.transform = transform
        self.messages_key = messages_key
        self.max_text_length = int(max_text_length)
        self.append_eos = bool(append_eos)
        self.eos_token_id = tokenizer.eos_token_id
        self.records: List[Dict[str, str]] = self._load_records(jsonl_path)

    # ─── 读取 ─────────────────────────────────────────────────
    def _load_records(self, path: str) -> List[Dict[str, str]]:
        """逐行读取并只保留 3 个字段（避免把 `messages` 全量版留在内存）。"""
        records: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
                if self.messages_key not in row:
                    raise KeyError(
                        f"{path}:{lineno} 缺字段 {self.messages_key!r}；"
                        f"现有字段: {sorted(row)}")
                records.append({
                    "image": row["image"],
                    self.messages_key: row[self.messages_key],
                    "gt": str(row.get("gt", "")),
                })
        if not records:
            raise ValueError(f"{path} 为空")
        return records

    def __len__(self) -> int:
        return len(self.records)

    # ─── 单样本 ───────────────────────────────────────────────
    def __getitem__(self, index: int) -> Dict[str, object]:
        """返回 dict：pixel_values / input_ids / attention_mask / labels / gt_name。"""
        record = self.records[index]
        image_path = os.path.join(self.image_root, record["image"])
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"第 {index} 条图片不存在: {image_path}")

        with Image.open(image_path) as image:
            pixel_values = self.transform(image.convert("RGB"))

        messages = self._parse_messages(record[self.messages_key])
        prompt_ids, answer_ids = self._build_ids(messages)
        ids = prompt_ids + answer_ids
        if len(ids) > self.max_text_length:
            raise ValueError(
                f"第 {index} 条文本长度 {len(ids)} > max_text_length="
                f"{self.max_text_length}（{record['image']}）；调大 "
                "--max-text-length 或检查数据")

        labels = [IGNORE_INDEX] * len(prompt_ids) + list(answer_ids)
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "gt_name": record["gt"] or os.path.basename(
                os.path.dirname(record["image"])),
        }

    @staticmethod
    def _parse_messages(raw) -> List[Dict[str, str]]:
        """`messages_short` 可能是 JSON 字符串或已解析的 list。"""
        messages = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"messages 至少要有 user/assistant 两条: {messages!r}")
        if not isinstance(messages[-1], dict) or "content" not in messages[-1]:
            raise ValueError(f"最后一条 message 缺 content: {messages[-1]!r}")
        return messages

    def _build_ids(self, messages: Sequence[Dict[str, str]]) -> tuple:
        """prompt ids（到 assistant header）+ answer ids（含 eos）。"""
        prompt_ids = self.tokenizer.apply_chat_template(
            list(messages[:-1]), tokenize=True, add_generation_prompt=True)
        full_ids = self.tokenizer.apply_chat_template(
            list(messages), tokenize=True)

        prompt_ids = list(prompt_ids)
        full_ids = list(full_ids)
        if full_ids[:len(prompt_ids)] == prompt_ids:
            answer_ids = full_ids[len(prompt_ids):]
        else:
            # chat template 升级导致前缀不再逐位一致时的兜底：直接 tokenize answer。
            answer_ids = self.tokenizer(
                messages[-1]["content"], add_special_tokens=False)["input_ids"]
            answer_ids = list(answer_ids)

        answer_ids = self._truncate_at_eos(answer_ids)
        if not answer_ids:
            raise ValueError("answer 为空（assistant content 是空串？）")
        return prompt_ids, answer_ids

    def _truncate_at_eos(self, answer_ids: List[int]) -> List[int]:
        """answer 截到第一个 eos（含），丢弃模板尾部的换行 token；没有则补 eos。"""
        eos = self.eos_token_id
        if eos is None:
            return list(answer_ids)
        if eos in answer_ids:
            return list(answer_ids[:answer_ids.index(eos) + 1])
        if self.append_eos:
            return list(answer_ids) + [eos]
        return list(answer_ids)


# ═══════════════════════════════════════════════════════════════
# Collator —— 只做 B 张图的 right-padding（8 行展开在 model 内）
# ═══════════════════════════════════════════════════════════════

class EightBudgetCollator:
    """把 B 个样本 right-pad 成批；返回的 `input_ids` 不含 visual token。

    8 个预算的展开由 `VLM8Budget.forward` 完成（每图 V 行），所以这里
    **batch 维度仍然是图片数 B**，不是 B×8。
    """

    def __init__(self, tokenizer, ignore_index: int = IGNORE_INDEX):
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError(
                "tokenizer.pad_token_id 为 None；train.py 里应设 "
                "tokenizer.pad_token = tokenizer.eos_token")
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, object]:
        """Args: `SelfDistillSFTDataset` 样本列表. Returns: Trainer 的 batch dict。"""
        if not features:
            raise ValueError("collator 收到空 batch")
        batch_size = len(features)
        max_len = max(int(feature["input_ids"].numel()) for feature in features)

        input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full(
            (batch_size, max_len), self.ignore_index, dtype=torch.long)

        for row, feature in enumerate(features):
            length = int(feature["input_ids"].numel())
            input_ids[row, :length] = feature["input_ids"]
            attention_mask[row, :length] = feature["attention_mask"]
            labels[row, :length] = feature["labels"]

        return {
            "pixel_values": torch.stack(
                [feature["pixel_values"] for feature in features], dim=0),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "gt_name": [str(feature.get("gt_name", "")) for feature in features],
        }
