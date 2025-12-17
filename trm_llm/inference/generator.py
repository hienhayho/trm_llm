"""Inference for TRM-LLM

Generates predictions with recursive refinement and ACT
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import json

from trm_llm.utils.config import TRMLLMConfig
from trm_llm.data.tokenizer import ToolCallTokenizer


class TRMInference:
    """Inference engine for TRM-LLM

    Uses recursive refinement with adaptive computation time (ACT)
    to generate predictions efficiently
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: ToolCallTokenizer,
        config: TRMLLMConfig,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Trained TRMLLM model
            tokenizer: ToolCallTokenizer
            config: TRMLLMConfig
            device: Device for inference
        """
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Build tool name mapping from dataset (should be loaded from checkpoint)
        self.tool_id_to_name = {}

    def set_tool_mapping(self, tool_name_to_id: Dict[str, int]):
        """Set tool name mapping

        Args:
            tool_name_to_id: Dict mapping tool names to IDs
        """
        self.tool_id_to_name = {v: k for k, v in tool_name_to_id.items()}

    @torch.no_grad()
    def generate(
        self,
        user_query: str,
        tools_json: str,
        max_steps: Optional[int] = None,
        generate_params: bool = True,
        generate_response: bool = True,
        max_param_len: int = 64,
        max_response_len: int = 128
    ) -> Dict:
        """Generate prediction for a user query

        Args:
            user_query: User's question
            tools_json: JSON string with tool definitions
            max_steps: Maximum supervision steps (uses config value if None)
            generate_params: Whether to generate tool parameters
            generate_response: Whether to generate response text for direct_answer
            max_param_len: Maximum parameter sequence length
            max_response_len: Maximum response sequence length

        Returns:
            result: Dict with:
                - action: 'direct_answer' or 'tool_call'
                - tool_name: Name of selected tool (if tool_call)
                - tool_id: ID of selected tool (if tool_call)
                - tool_call: Full tool call dict with name and arguments (if tool_call)
                - response: Generated response text (if direct_answer)
                - confidence: Confidence score
                - num_steps: Number of refinement steps used
        """
        max_steps = max_steps or self.config.max_supervision_steps

        # Encode input
        messages = [{'role': 'user', 'content': user_query}]
        input_ids = self.tokenizer.encode_conversation(tools_json, messages)
        input_ids = torch.tensor([input_ids], device=self.device)

        # Forward with ACT (early stopping)
        outputs_per_step = self.model(
            input_ids,
            max_supervision_steps=max_steps,
            training=False  # Enables ACT early stopping
        )

        num_steps_used = len(outputs_per_step)

        # Use final output (or early stopped output)
        final_output = outputs_per_step[-1]

        # Decode action
        action_probs = F.softmax(final_output['action_logits'], dim=-1)[0]
        action_id = action_probs.argmax().item()

        if action_id == 0:
            # Direct answer - generate response text
            response_text = None
            if generate_response and hasattr(self.model, 'generate_response'):
                y_state = final_output['y_state']  # (1, action_dim)
                response_ids = self.model.generate_response(y_state, max_length=max_response_len)
                response_text = self.tokenizer.decode(response_ids[0].tolist(), skip_special_tokens=True)

            return {
                'action': 'direct_answer',
                'tool_name': None,
                'tool_id': None,
                'num_parallel_calls': 0,
                'tool_call': None,
                'response': response_text,
                'confidence': action_probs[0].item(),
                'num_steps': num_steps_used
            }
        else:
            # Tool call
            tool_probs = F.softmax(final_output['tool_logits'], dim=-1)[0]
            tool_id = tool_probs.argmax().item()
            tool_name = self.tool_id_to_name.get(tool_id, f"unknown_tool_{tool_id}")

            # Number of parallel calls
            num_parallel_calls = 1
            if 'num_calls_logits' in final_output:
                num_calls_probs = F.softmax(final_output['num_calls_logits'], dim=-1)[0]
                num_parallel_calls = num_calls_probs.argmax().item() + 1  # 0-indexed to 1-indexed

            confidence = action_probs[1].item() * tool_probs[tool_id].item()

            # Generate tool parameters
            tool_call = None
            if generate_params and hasattr(self.model, 'generate_params'):
                y_state = final_output['y_state']  # (1, action_dim)
                param_ids = self.model.generate_params(y_state, max_length=max_param_len)
                param_text = self.tokenizer.decode(param_ids[0].tolist(), skip_special_tokens=True)
                tool_call = self._parse_tool_call(param_text, tool_name)

            return {
                'action': 'tool_call',
                'tool_name': tool_name,
                'tool_id': tool_id,
                'num_parallel_calls': num_parallel_calls,
                'tool_call': tool_call,
                'response': None,
                'confidence': confidence,
                'num_steps': num_steps_used
            }

    def _parse_tool_call(self, param_text: str, fallback_name: str) -> Optional[Dict]:
        """Parse generated parameter text into tool call dict

        Args:
            param_text: Generated text (hopefully JSON)
            fallback_name: Tool name to use if parsing fails

        Returns:
            tool_call: Dict with 'name' and 'arguments', or None if parsing fails
        """
        # Try to extract JSON from the generated text
        param_text = param_text.strip()

        # Try to find JSON object in the text
        start_idx = param_text.find('{')
        end_idx = param_text.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = param_text[start_idx:end_idx + 1]
            try:
                parsed = json.loads(json_str)
                # Check if it's a full tool call or just arguments
                if 'name' in parsed and 'arguments' in parsed:
                    return parsed
                elif 'arguments' in parsed:
                    return {'name': fallback_name, 'arguments': parsed['arguments']}
                else:
                    # Assume the whole thing is arguments
                    return {'name': fallback_name, 'arguments': parsed}
            except json.JSONDecodeError:
                pass

        # Return None if parsing fails
        return {'name': fallback_name, 'arguments': {}, 'raw_output': param_text}

    @torch.no_grad()
    def generate_batch(
        self,
        user_queries: List[str],
        tools_json: str,
        max_steps: Optional[int] = None
    ) -> List[Dict]:
        """Generate predictions for a batch of queries

        Args:
            user_queries: List of user questions
            tools_json: JSON string with tool definitions
            max_steps: Maximum supervision steps

        Returns:
            results: List of result dicts
        """
        max_steps = max_steps or self.config.max_supervision_steps

        # Encode all queries
        all_input_ids = []
        for query in user_queries:
            messages = [{'role': 'user', 'content': query}]
            input_ids = self.tokenizer.encode_conversation(tools_json, messages)
            all_input_ids.append(input_ids)

        # Pad to same length
        max_len = max(len(ids) for ids in all_input_ids)
        padded_ids = []
        for ids in all_input_ids:
            padding = [self.tokenizer.pad_token_id] * (max_len - len(ids))
            padded_ids.append(ids + padding)

        input_ids = torch.tensor(padded_ids, device=self.device)

        # Forward pass
        outputs_per_step = self.model(
            input_ids,
            max_supervision_steps=max_steps,
            training=False
        )

        num_steps_used = len(outputs_per_step)
        final_output = outputs_per_step[-1]

        # Decode actions for all examples
        batch_size = input_ids.size(0)
        results = []

        for i in range(batch_size):
            action_probs = F.softmax(final_output['action_logits'][i], dim=-1)
            action_id = action_probs.argmax().item()

            if action_id == 0:
                results.append({
                    'action': 'direct_answer',
                    'tool_name': None,
                    'tool_id': None,
                    'confidence': action_probs[0].item(),
                    'num_steps': num_steps_used
                })
            else:
                tool_probs = F.softmax(final_output['tool_logits'][i], dim=-1)
                tool_id = tool_probs.argmax().item()
                tool_name = self.tool_id_to_name.get(tool_id, f"unknown_tool_{tool_id}")

                results.append({
                    'action': 'tool_call',
                    'tool_name': tool_name,
                    'tool_id': tool_id,
                    'confidence': action_probs[1].item() * tool_probs[tool_id].item(),
                    'num_steps': num_steps_used
                })

        return results

    def analyze_refinement(
        self,
        user_query: str,
        tools_json: str
    ) -> Dict:
        """Analyze how the model refines its prediction across steps

        Args:
            user_query: User's question
            tools_json: Tool definitions

        Returns:
            analysis: Dict with per-step predictions and confidences
        """
        messages = [{'role': 'user', 'content': user_query}]
        input_ids = self.tokenizer.encode_conversation(tools_json, messages)
        input_ids = torch.tensor([input_ids], device=self.device)

        # Forward with all steps
        outputs_per_step = self.model(
            input_ids,
            max_supervision_steps=self.config.max_supervision_steps,
            training=False
        )

        # Analyze each step
        step_analysis = []

        for step_idx, outputs in enumerate(outputs_per_step):
            action_probs = F.softmax(outputs['action_logits'][0], dim=-1)
            tool_probs = F.softmax(outputs['tool_logits'][0], dim=-1)
            halt_prob = torch.sigmoid(outputs['halt_logit'][0]).item()

            action_id = action_probs.argmax().item()
            tool_id = tool_probs.argmax().item()

            step_analysis.append({
                'step': step_idx + 1,
                'action': 'tool_call' if action_id == 1 else 'direct_answer',
                'action_confidence': action_probs[action_id].item(),
                'tool_id': tool_id if action_id == 1 else None,
                'tool_name': self.tool_id_to_name.get(tool_id) if action_id == 1 else None,
                'tool_confidence': tool_probs[tool_id].item() if action_id == 1 else None,
                'halt_prob': halt_prob,
            })

        return {
            'query': user_query,
            'steps': step_analysis,
            'total_steps': len(step_analysis)
        }
