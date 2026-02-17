# MedGemma HF Endpoint Reference

## Endpoint

- **URL**: `https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud`
- **Model**: `google/medgemma-1-5-4b-it-hae` (multimodal, image-text-to-text)
- **Provider**: HuggingFace Inference Endpoints (AWS us-east-1)
- **Runtime**: Default HF inference image (**NOT** TGI)

## Authentication

- Requires `HF_TOKEN` environment variable (HuggingFace access token)
- Token must have access to the endpoint

## API Constraints

| Feature | Status | Notes |
|---------|--------|-------|
| `/v1/chat/completions` | 404 | NOT available — default image, not TGI |
| `text_generation()` | Works | Must use Gemma chat template manually |
| Multimodal (images) | Works | Base64 encode, extract text parts for prompt |
| Streaming | Not tested | Use non-streaming `text_generation()` |

## Gemma Chat Template

The endpoint requires manual chat template formatting:

```
<start_of_turn>user
[system prompt folded here]

[user message]<end_of_turn>
<start_of_turn>model
[response generated here]
```

Key rules:
- System prompt is folded into the first user turn (Gemma has no system role)
- Each turn wrapped in `<start_of_turn>role\n...<end_of_turn>`
- Final `<start_of_turn>model\n` opens generation

## Cold-Start Behavior

The endpoint may return **503** when scaling from zero. Retry strategy:

| Setting | Default | Env Var |
|---------|---------|---------|
| Max retries | 6 | `MEDGEMMA_MAX_RETRIES` |
| Backoff base | 2.0s | `MEDGEMMA_RETRY_BACKOFF` |
| Max per-retry wait | 60s | `MEDGEMMA_MAX_WAIT` |
| Total retry budget | 60s | `MEDGEMMA_COLD_START_TIMEOUT` |

Backoff formula: `min(backoff_base ** attempt, max_wait)`

## Output Parsing

MedGemma may wrap JSON output in markdown code blocks:

```
\`\`\`json
{"key": "value"}
\`\`\`
```

Always use `parse_json()` which handles both raw JSON and markdown-wrapped JSON.

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| 503 Service Unavailable | Cold start | Retry with backoff (handled by client) |
| 404 on chat/completions | Using TGI API on non-TGI endpoint | Use `text_generation()` with Gemma template |
| 401 Unauthorized | Bad/missing HF_TOKEN | Check token and endpoint access |
| Empty response | Model returned nothing | Retry or increase `max_new_tokens` |
| JSON parse error | Markdown-wrapped output | Use `parse_json()` helper |
