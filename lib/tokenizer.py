"""Tokenizer helpers with chat template fallbacks."""

from transformers import AutoTokenizer

# Gemma 4 models may not ship with a chat_template in the tokenizer config.
GEMMA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
    "{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
    "{% elif message['role'] == 'system' %}<start_of_turn>system\n{{ message['content'] }}<end_of_turn>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


def setup_tokenizer(model_name, chat_template_fallback=GEMMA_CHAT_TEMPLATE):
    """Load tokenizer, set pad_token, and apply chat template fallback if needed."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, 'chat_template', None) and chat_template_fallback:
        tokenizer.chat_template = chat_template_fallback
    return tokenizer
