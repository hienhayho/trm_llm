"""Data collator for batching tool-calling examples"""

import torch
from typing import Dict, List


class DataCollator:
    """Collate tool-calling examples into batches

    Handles padding of variable-length sequences
    """

    def __init__(self, pad_token_id: int):
        """
        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples

        Args:
            batch: List of dicts from dataset, each with:
                - input_ids: List[int]
                - target_action: int
                - target_tool_id: int
                - target_num_calls: int
                - target_generation_ids: List[int] (may be empty)

        Returns:
            batched: Dict with:
                - input_ids: (batch_size, max_len)
                - target_action: (batch_size,)
                - target_tool_id: (batch_size,)
                - target_num_calls: (batch_size,)
                - attention_mask: (batch_size, max_len)
                - target_generation_ids: (batch_size, max_gen_len)
                - generation_mask: (batch_size, max_gen_len)
        """
        # Find max length in batch
        max_len = max(len(item['input_ids']) for item in batch)

        # Find max generation length
        gen_lengths = [len(item.get('target_generation_ids', [])) for item in batch]
        max_gen_len = max(gen_lengths) if any(gen_lengths) else 1  # At least 1 for tensor shape

        # Pad sequences
        input_ids = []
        attention_masks = []
        target_generation_ids = []
        generation_masks = []

        for item in batch:
            ids = item['input_ids']
            seq_len = len(ids)

            # Padding for input
            padding_len = max_len - seq_len
            padded_ids = ids + [self.pad_token_id] * padding_len

            # Attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * seq_len + [0] * padding_len

            input_ids.append(padded_ids)
            attention_masks.append(attention_mask)

            # Padding for generation ids
            gen_ids = item.get('target_generation_ids', [])
            gen_len = len(gen_ids)
            gen_padding_len = max_gen_len - gen_len
            padded_gen_ids = gen_ids + [self.pad_token_id] * gen_padding_len
            gen_mask = [1] * gen_len + [0] * gen_padding_len

            target_generation_ids.append(padded_gen_ids)
            generation_masks.append(gen_mask)

        # Convert to tensors
        batched = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'target_action': torch.tensor([item['target_action'] for item in batch], dtype=torch.long),
            'target_tool_id': torch.tensor([item['target_tool_id'] for item in batch], dtype=torch.long),
            'target_num_calls': torch.tensor([item.get('target_num_calls', 0) for item in batch], dtype=torch.long),
            'target_generation_ids': torch.tensor(target_generation_ids, dtype=torch.long),
            'generation_mask': torch.tensor(generation_masks, dtype=torch.long),
        }

        return batched


class DataCollatorWithPadding:
    """Alternative collator with better padding strategy

    Pads to multiples of 8 for better GPU utilization
    """

    def __init__(self, pad_token_id: int, padding_multiple: int = 8):
        """
        Args:
            pad_token_id: Token ID for padding
            padding_multiple: Pad to multiples of this number
        """
        self.pad_token_id = pad_token_id
        self.padding_multiple = padding_multiple

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with efficient padding"""
        # Find max length
        max_len = max(len(item['input_ids']) for item in batch)

        # Round up to nearest multiple
        if self.padding_multiple > 1:
            max_len = ((max_len + self.padding_multiple - 1) // self.padding_multiple) * self.padding_multiple

        # Pad sequences
        input_ids = []
        attention_masks = []

        for item in batch:
            ids = item['input_ids']
            seq_len = len(ids)
            padding_len = max_len - seq_len

            # Pad
            padded_ids = ids + [self.pad_token_id] * padding_len
            attention_mask = [1] * seq_len + [0] * padding_len

            input_ids.append(padded_ids)
            attention_masks.append(attention_mask)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'target_action': torch.tensor([item['target_action'] for item in batch], dtype=torch.long),
            'target_tool_id': torch.tensor([item['target_tool_id'] for item in batch], dtype=torch.long),
        }
