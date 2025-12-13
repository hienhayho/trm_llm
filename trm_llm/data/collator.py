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
                - target_param_ids: List[int] (may be empty)
                - target_response_ids: List[int] (may be empty)

        Returns:
            batched: Dict with:
                - input_ids: (batch_size, max_len)
                - target_action: (batch_size,)
                - target_tool_id: (batch_size,)
                - attention_mask: (batch_size, max_len)
                - target_param_ids: (batch_size, max_param_len)
                - param_mask: (batch_size, max_param_len)
                - target_response_ids: (batch_size, max_response_len)
                - response_mask: (batch_size, max_response_len)
        """
        # Find max length in batch
        max_len = max(len(item['input_ids']) for item in batch)

        # Find max param length (only for items with tool calls)
        param_lengths = [len(item.get('target_param_ids', [])) for item in batch]
        max_param_len = max(param_lengths) if any(param_lengths) else 1  # At least 1 for tensor shape

        # Find max response length (only for direct_answer samples)
        response_lengths = [len(item.get('target_response_ids', [])) for item in batch]
        max_response_len = max(response_lengths) if any(response_lengths) else 1  # At least 1 for tensor shape

        # Pad sequences
        input_ids = []
        attention_masks = []
        target_param_ids = []
        param_masks = []
        target_response_ids = []
        response_masks = []

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

            # Padding for param ids
            param_ids = item.get('target_param_ids', [])
            param_len = len(param_ids)
            param_padding_len = max_param_len - param_len
            padded_param_ids = param_ids + [self.pad_token_id] * param_padding_len
            param_mask = [1] * param_len + [0] * param_padding_len

            target_param_ids.append(padded_param_ids)
            param_masks.append(param_mask)

            # Padding for response ids
            response_ids = item.get('target_response_ids', [])
            response_len = len(response_ids)
            response_padding_len = max_response_len - response_len
            padded_response_ids = response_ids + [self.pad_token_id] * response_padding_len
            response_mask = [1] * response_len + [0] * response_padding_len

            target_response_ids.append(padded_response_ids)
            response_masks.append(response_mask)

        # Convert to tensors
        batched = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'target_action': torch.tensor([item['target_action'] for item in batch], dtype=torch.long),
            'target_tool_id': torch.tensor([item['target_tool_id'] for item in batch], dtype=torch.long),
            'target_num_calls': torch.tensor([item.get('target_num_calls', 0) for item in batch], dtype=torch.long),
            'target_param_ids': torch.tensor(target_param_ids, dtype=torch.long),
            'param_mask': torch.tensor(param_masks, dtype=torch.long),
            'target_response_ids': torch.tensor(target_response_ids, dtype=torch.long),
            'response_mask': torch.tensor(response_masks, dtype=torch.long),
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
