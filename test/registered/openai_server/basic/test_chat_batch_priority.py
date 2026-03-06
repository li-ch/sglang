import unittest
import uuid
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionBatchRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    MessageProcessingResult,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")


class _MockTokenizerManager:
    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
            context_length=None,
            allow_auto_truncate=True,
        )
        mock_hf_config = Mock()
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        self.model_config.hf_config = mock_hf_config
        self.chat_template_name: Optional[str] = "llama-3"
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2]
        self.tokenizer.bos_token_id = 1

        async def _mock_generate():
            yield {
                "text": "ok",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()


class _MockTemplateManager:
    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None


class TestChatBatchPriority(unittest.TestCase):
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)
        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    def _mock_message_processing(self):
        return MessageProcessingResult(
            prompt="Test prompt",
            prompt_ids=[1, 2],
            image_data=None,
            video_data=None,
            audio_data=None,
            stop=["</s>"],
            tool_call_constraint=None,
        )

    def test_priority_precedence(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=False,
            sampling_params={"priority": 5},
            priority=3,
        )
        self.fastapi_request.headers = {"x-priority": "7"}
        with patch.object(self.chat, "_process_messages") as proc_mock:
            proc_mock.return_value = self._mock_message_processing()
            adapted, _ = self.chat._convert_to_internal_request(req, self.fastapi_request)
        self.assertIsInstance(adapted, GenerateReqInput)
        self.assertEqual(adapted.priority, 3)

    def test_priority_falls_back_to_sampling_params_then_header(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=False,
            sampling_params={"priority": 5},
            priority=None,
        )
        self.fastapi_request.headers = {"x-priority": "7"}
        with patch.object(self.chat, "_process_messages") as proc_mock:
            proc_mock.return_value = self._mock_message_processing()
            adapted, _ = self.chat._convert_to_internal_request(req, self.fastapi_request)
        self.assertEqual(adapted.priority, 5)

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=False,
            sampling_params=None,
            priority=None,
        )
        with patch.object(self.chat, "_process_messages") as proc_mock:
            proc_mock.return_value = self._mock_message_processing()
            adapted, _ = self.chat._convert_to_internal_request(req, self.fastapi_request)
        self.assertEqual(adapted.priority, 7)

    def test_batch_endpoint_non_streaming(self):
        batch = ChatCompletionBatchRequest(
            requests=[
                ChatCompletionRequest(
                    model="x",
                    messages=[{"role": "user", "content": "Hi?"}],
                    stream=False,
                )
            ]
        )

        async def _fake_non_streaming(*_args, **_kwargs):
            return ChatCompletionResponse(
                id="chatcmpl-test",
                created=123,
                model="x",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        loop = get_or_create_event_loop()
        with patch.object(self.chat, "_process_messages") as proc_mock, patch.object(
            self.chat, "_handle_non_streaming_request", side_effect=_fake_non_streaming
        ):
            proc_mock.return_value = self._mock_message_processing()
            responses = loop.run_until_complete(
                self.chat.handle_batch_request(batch, self.fastapi_request)
            )
        self.assertEqual(len(responses), 1)


if __name__ == "__main__":
    unittest.main(verbosity=3)
