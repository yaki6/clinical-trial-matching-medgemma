# Deep Research Report: MedGemma Official Tutorials, Notebooks, and Best Practices

**Date**: 2026-02-23
**Researcher**: deep-research-agent (tutorial-researcher role)
**Repositories Analyzed**: Google-Health/medgemma, vllm-project/vllm
**Total Research Rounds**: 4

---

## Executive Summary

MedGemma 4B官方部署途径有两条：HuggingFace Transformers本地推理和Vertex AI Model Garden云端部署。官方笔记本(`quick_start_with_model_garden.ipynb`)明确展示了Vertex AI上的OpenAI兼容API用法，图像通过`image_url`字段以URL形式传递（官方示例不使用base64），文本在图像前出现于消息体中。关键发现：(1) 官方推荐temperature=0做医疗图像分析；(2) SigLIP图像编码器内部将图像resize到896x896，调用者无需手动预处理；(3) vLLM Vertex AI部署使用`--max-model-len=32768`而非128K；(4) MedGemma 4B原版(v1.0.0)存在end-of-image token缺失的严重bug，2025年7月9日通过v1.0.1修复，影响所有combined text+image任务；(5) 系统消息不是必须的，研究表明其对domain-specific MedGemma性能影响不显著，但官方示例均包含系统消息。

---

## Research Objectives

1. 找到MedGemma 4B多模态推理的官方Google/HuggingFace笔记本
2. 了解Vertex AI Model Garden MedGemma部署指南
3. 掌握MedGemma 4B图像任务的prompt工程最佳实践
4. 明确SigLIP图像预处理要求（格式、尺寸、归一化）
5. 了解vLLM serving特殊配置需求

---

## Detailed Findings

### Opening Item 1: 官方笔记本和代码示例

#### Round 1: Surface Exploration

**Questions Asked**: MedGemma官方repo结构是什么？有哪些官方笔记本？

**Key Discoveries**:
- 官方repo: `https://github.com/Google-Health/medgemma`
- 提供三个核心笔记本：
  - `quick_start_with_hugging_face.ipynb` — 本地HF Transformers推理
  - `quick_start_with_model_garden.ipynb` — Vertex AI云端部署和推理
  - `fine_tune_with_hugging_face.ipynb` — LoRA微调示例
- HuggingFace hub-tutorials还有另一个参考笔记本：`05-medgemma-1.5.ipynb`

**Initial Gaps**: 笔记本的具体代码细节，特别是图像传递格式

#### Round 2: Deep Dive

**Questions Asked**: `quick_start_with_model_garden.ipynb`中的多模态推理代码细节？OpenAI兼容API的消息结构？

**Key Discoveries**:

**Vertex AI SDK方式（使用`<start_of_image>`特殊token）**:
```python
formatted_prompt = "You are an expert radiologist. Describe this X-ray <start_of_image>"

instances = [
    {
        "prompt": formatted_prompt,
        "multi_modal_data": {"image": image_url},  # 公开URL
        "max_tokens": 200,
        "temperature": 0,
        "raw_response": True,
    },
]

response = endpoints["endpoint"].predict(
    instances=instances, use_dedicated_endpoint=True
)
```

**OpenAI SDK方式（推荐，无需特殊token）**:
```python
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist"}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this X-ray"},        # 文本在前
            {"type": "image_url", "image_url": {"url": image_url}}  # 图像在后
        ]
    }
]

model_response = client.chat.completions.create(
    model="",           # Vertex AI端点用空字符串
    messages=messages,
    max_tokens=300,     # 图像任务推荐200-300
    temperature=0,      # 医疗任务推荐deterministic
)
```

**关键区别**:
- Vertex AI SDK的`predict()`方法需要`<start_of_image>`特殊token在prompt字符串中
- OpenAI SDK的`chat.completions.create()`**不需要**`<start_of_image>`token，由API自动处理
- 官方示例中图像以公开URL形式传递，**不使用base64**

**Emerging Patterns**:
- 官方笔记本使用`temperature=0`做确定性生成
- 图像token数量固定：每张图像编码为256个token
- 系统消息采用专家角色扮演模式（如"You are an expert radiologist"）

#### Round 3: Crystallization

**Questions Asked**: 消息体中文本和图像的顺序？BASE_URL如何构造？

**Final Understanding**:

在OpenAI SDK方式中，`user`的`content`数组中**文本在前，图像在后**：
```python
"content": [
    {"type": "text", "text": "Describe this X-ray"},        # 先文本
    {"type": "image_url", "image_url": {"url": image_url}}  # 后图像
]
```

这与HuggingFace fine-tune笔记本的格式相反（HF格式中**图像在前，文本在后**），这是Vertex AI OpenAI兼容API的特定格式。

**BASE_URL构造**（两种模式）:
```python
# Dedicated Endpoint（推荐）
DEDICATED_ENDPOINT_DNS = endpoints["endpoint"].gca_resource.dedicated_endpoint_dns
BASE_URL = f"https://{DEDICATED_ENDPOINT_DNS}/v1beta1/{ENDPOINT_RESOURCE_NAME}"

# Standard Endpoint（备用）
BASE_URL = f"https://{ENDPOINT_REGION}-aiplatform.googleapis.com/v1beta1/{ENDPOINT_RESOURCE_NAME}"
```

**Validated Assumptions**:
- `model=""` (空字符串) 是Vertex AI端点的正确用法
- `use_dedicated_endpoint=True` 是Model Garden部署的默认设置

---

### Opening Item 2: Vertex AI Model Garden部署指南

#### Round 1: Surface Exploration

**Questions Asked**: MedGemma如何在Vertex AI Model Garden部署？有哪些方式？

**Key Discoveries**:
- 两种主要部署方式：
  1. 通过Google Cloud Console Model Garden界面一键部署（在线预测）
  2. 通过`aiplatform.Model.upload()`上传模型到注册中心（批量预测）
- 两种推理模式：
  - Online Predictions — 同步、低延迟，需要dedicated endpoint
  - Batch Predictions — 异步，适合大规模数据

**Initial Gaps**: 具体的vLLM配置参数

#### Round 2: Deep Dive

**Questions Asked**: `quick_start_with_model_garden.ipynb`中的vLLM配置？机器类型？GPU数量？

**Key Discoveries**:

**MedGemma 4B官方推荐配置**:
```
机器类型: g2-standard-24
GPU类型: NVIDIA_L4 x 2
```

**vLLM启动参数**:
```python
vllm_args = [
    f"--model={model_id}",                     # google/medgemma-4b-it
    f"--tensor-parallel-size=2",               # 匹配GPU数量
    f"--max-model-len=32768",                  # 注意：32K，不是128K！
    f"--gpu-memory-utilization=0.95",
    f"--max-num-seqs=4",
    f"--swap-space=16",
    "--disable-log-stats",
    "--host=0.0.0.0",
    "--port=8080",
]
```

**重要发现**: 官方notebook使用`--max-model-len=32768`（32K tokens），而不是营销材料中提到的128K上下文。这是因为2x L4 VRAM限制。

**Serving Docker Image**:
```
pytorch-vllm-serve:20250430_0916_RC00_maas
```

#### Round 3: Crystallization

**Questions Asked**: 访问Model Garden endpoint需要什么认证？

**Final Understanding**:

```python
import google.auth
import google.auth.transport.requests
import openai

# 获取Google Cloud认证
creds, project = google.auth.default()
auth_req = google.auth.transport.requests.Request()
creds.refresh(auth_req)

# 配置OpenAI客户端
client = openai.OpenAI(base_url=BASE_URL, api_key=creds.token)
```

使用Google Cloud Application Default Credentials，`api_key`字段填入`creds.token`（不是传统的API密钥）。

---

### Opening Item 3: 图像预处理要求

#### Round 1: Surface Exploration

**Questions Asked**: SigLIP期望什么图像格式和尺寸？

**Key Discoveries**:
- SigLIP图像编码器在内部将图像resize到**896x896**像素
- 像素值归一化到[-1, 1]范围
- 输出：每张图像编码为**256个token**
- MedSigLIP（公开版）使用448x448；MedGemma 4B集成版使用896x896

**Initial Gaps**: 调用者是否需要手动预处理？非正方形图像如何处理？

#### Round 2: Deep Dive

**Questions Asked**: CT图像的特殊处理？Pan&Scan算法细节？

**Key Discoveries**:

**CT图像特殊处理**（来自MedGemma Technical Report）:
```
CT图像 → 三窗口处理 → RGB三通道
- 通道1: 骨/肺窗 (bone/lung window)
- 通道2: 软组织窗 (soft tissue window)
- 通道3: 脑窗 (brain window)
每个窗口有特定的 window-width 和 window-level 参数
```

**高分辨率/非正方形图像**（Pan&Scan算法）:
```
1. 自适应裁剪图像
2. 将每个crop resize到896x896
3. 分别编码每个crop
```

**调用者责任**:
- 当使用HuggingFace Transformers的`AutoProcessor`时：**无需手动预处理**，处理器自动完成
- 当使用Vertex AI API（URL传递）时：同样无需手动预处理，模型服务端处理
- 仅当绕过官方API直接使用vLLM时，才需要关注预处理细节

#### Round 3: Crystallization

**Questions Asked**: base64编码图像是否有问题？格式要求？

**Final Understanding**:

1. **官方API路径（推荐）**: 图像以公开URL传递，服务端处理所有预处理
   ```python
   {"type": "image_url", "image_url": {"url": "https://..."}}
   ```

2. **base64路径**: vLLM在某些版本中存在base64图像处理bug（PIL.UnidentifiedImageError），已在2025年11月修复。格式应为：
   ```python
   f"data:image/jpeg;base64,{base64_encoded_image}"
   ```

3. **重要bug**: MedGemma 4B-IT v1.0.0存在**end-of-image token缺失**的严重bug，导致多模态性能下降。该bug在**2025年7月9日v1.0.1**中修复。如果使用的是修复前的模型权重，多模态性能会受到显著影响。

**图像质量建议**（来自研究经验）:
- 医疗图像（胸部X光、皮肤科图像、眼底图像、组织病理切片）是SigLIP的训练域
- 自然图像也被支持
- 彩色JPEG/PNG格式工作正常

---

### Opening Item 4: Prompt工程最佳实践

#### Round 1: Surface Exploration

**Questions Asked**: MedGemma 4B的推荐prompt格式是什么？

**Key Discoveries**:
- MedGemma官方文档明确警告：**"MedGemma对特定prompt的敏感性高于Gemma 3"**
- 官方示例使用专家角色扮演系统消息（如"You are an expert radiologist"）
- 推荐使用结构化输出请求（如"Provide: DIAGNOSIS and FINDINGS"）

**Initial Gaps**: 系统消息是否真的有帮助？chain-of-thought是否有效？

#### Round 2: Deep Dive

**Questions Asked**: 系统消息对MedGemma性能的影响？CoT推理？

**Key Discoveries**:

**系统消息研究结果（来自MEDIQA-OE 2025论文）**:
- 研究人员给通用模型提供系统prompt，但对MedGemma等domain-specific模型不提供
- 发现："This change does not significantly influence the performance of the models"
- 结论：对于MedGemma这类医学专用模型，系统消息的影响不显著

**CoT推理研究结果（来自医学LLM鲁棒性研究）**:
- MedGemma 4B展示了"chain-of-thought prompting的异常行为"
- CoT prompting (P1) 将准确率从73.1%降至56.2%（下降16.9个百分点！）
- Self-consistency voting (P2) 可以恢复性能至74.3%
- **结论**: 直接CoT可能损害MedGemma 4B的性能，需谨慎使用

**MedGemma 1.5 Thinking Mode**（新特性）:
- MedGemma 1.5支持thinking mode
- 对于文本任务，thinking mode可能改善性能
- 代价是生成更多token（max_completion_tokens需要设置更大，如1500）

#### Round 3: Crystallization

**Questions Asked**: 图像任务的最佳prompt模板？哪些格式被证明有效？

**Final Understanding**:

**有效的prompt结构**（基于官方示例和社区验证）:

```python
# 方式1：简单直接（官方Model Garden notebook示例）
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this X-ray"},  # 文本在前
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }
]

# 方式2：结构化输出请求
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Based on the image and clinical history below, provide:\n"
                        "DIAGNOSIS: [primary diagnosis]\n"
                        "FINDINGS: [key findings]\n\n"
                        "Clinical History: [patient history here]"
            },
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }
]
```

**关键设计原则**:
1. **文本在前，图像在后**（Vertex AI OpenAI API格式）
2. **temperature=0** 用于医疗分析，确保确定性输出
3. **max_tokens=200-300** 适合图像描述任务；文本任务可用500
4. **不使用CoT**（直接CoT会降低MedGemma 4B性能）
5. 系统消息影响不显著，但官方示例包含，可保留

---

### Opening Item 5: vLLM Serving特殊配置

#### Round 1: Surface Exploration

**Questions Asked**: MedGemma on vLLM有什么特殊配置需求？

**Key Discoveries**:
- Vertex AI使用定制版vLLM，不是开源原版
- 特别针对Vertex AI input/output格式和AIP_*环境变量做了兼容性调整
- 推荐Docker镜像：`pytorch-vllm-serve:20250430_0916_RC00_maas`

**Initial Gaps**: 完整的vLLM参数清单

#### Round 2: Deep Dive

**Questions Asked**: MedGemma 4B的完整vLLM参数配置？

**Key Discoveries**:

**完整vLLM参数（官方batch prediction notebook）**:
```python
vllm_args = [
    "--model=google/medgemma-4b-it",
    "--tensor-parallel-size=2",    # 必须 = GPU数量
    "--max-model-len=32768",        # 32K (受VRAM限制)
    "--gpu-memory-utilization=0.95",
    "--max-num-seqs=4",             # 最大并发序列
    "--swap-space=16",
    "--disable-log-stats",
    "--host=0.0.0.0",
    "--port=8080",
]
```

**MedGemma 27B对比配置**:
```python
# MedGemma 27B需要8x L4 GPUs
# Machine: g2-standard-96
# --tensor-parallel-size=8
# --max-model-len=32768
```

#### Round 3: Crystallization

**Questions Asked**: 有没有额外的性能优化标志？

**Final Understanding**:

Google Cloud文档中提到的可选优化标志（针对Gemma/MedGemma）:
```
--enable-chunked-prefill    # 优化长文本prefill
--enable-prefix-caching     # 优化重复前缀（多轮对话）
```

**关于max_model_len**: 营销材料说128K，但官方notebook用32768。这是因为：
- 2x L4 (48GB total VRAM) - 模型权重~7GB (4B int8) - 其余用于KV cache
- 32K token长度是在该硬件配置下的实际上限

---

## Cross-Cutting Insights

1. **API格式二元性**: Vertex AI SDK格式 vs OpenAI SDK格式存在微妙差异：
   - Vertex SDK需要`<start_of_image>` token在prompt字符串中
   - OpenAI SDK无需特殊token，图像通过`image_url`自动处理
   - **当前我们的实现使用OpenAI格式，这是正确的**

2. **图像顺序不一致**:
   - HuggingFace本地格式：图像在前，文本在后
   - Vertex AI OpenAI格式：文本在前，图像在后
   - 两种方式都被Google官方文档使用，差异来自不同的底层API

3. **end-of-image token bug**: MedGemma 4B v1.0.0（原版）有严重的多模态性能问题。v1.0.1修复了这个bug。如果我们部署的是Model Garden的最新版本，应该已包含修复。

4. **系统消息争议**: 官方示例都包含系统消息，但研究表明其对domain-specific模型影响不显著。保留系统消息作为"专家角色扮演"是安全的做法。

---

## Architecture/Design Decisions

### 决策1: 使用OpenAI兼容API格式

**理由**: Vertex AI Model Garden的OpenAI兼容接口(`/v1beta1/{endpoint}`)是官方推荐的生产路径，无需特殊token处理。

**权衡**:
- 优势：无需`<start_of_image>`token，标准格式，易移植
- 劣势：相比直接Vertex SDK调用多了一层抽象

**官方代码示例**:
```python
client = openai.OpenAI(base_url=BASE_URL, api_key=creds.token)
response = client.chat.completions.create(
    model="",
    messages=messages,
    max_tokens=200,
    temperature=0,
)
```

### 决策2: 图像以URL形式传递 vs base64

**官方推荐**: URL形式。官方notebook只展示了URL传递。

**base64的问题**:
- vLLM存在过base64处理bug（已修复，但需要最新版本）
- base64数据会增加请求大小
- Vertex AI的定制vLLM是否完全支持base64尚不明确

**如果必须使用base64**（如本地图像文件）:
```python
import base64
with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")
image_url = f"data:image/jpeg;base64,{img_b64}"
```

---

## Edge Cases & Limitations

1. **多图像支持**: MedGemma 4B经过评估的主要是单图像任务，多图像理解场景未经充分评估

2. **多轮对话**: 官方文档明确说明"MedGemma未针对multi-turn应用进行评估和优化"

3. **CT图像**: 需要三窗口预处理才能发挥最佳性能，原始DICOM格式需要特别处理

4. **max_new_tokens限制**: 虽然上下文支持32K，但生成长度受限于GPU VRAM，实际推荐max_tokens=200-300

5. **Chain-of-thought**: 对MedGemma 4B有反效果（-16.9pp准确率），避免使用显式CoT指令

6. **v1.0.0 end-of-image bug**: 如果使用的是2025年7月9日之前的模型权重，多模态性能会显著下降

---

## Recommendations

基于所有研究发现，以下是对当前实现的建议：

### 1. 图像顺序修正（优先级：中）
当前prompt可能是"图像在前，文本在后"。根据Vertex AI OpenAI API的官方示例，应该是"文本在前，图像在后"：
```python
"content": [
    {"type": "text", "text": "...prompt..."},       # 文本在前
    {"type": "image_url", "image_url": {"url": ...}} # 图像在后
]
```

### 2. 简化prompt（优先级：高）
研究确认CoT会降低MedGemma 4B性能。当前的结构化输出prompt（"Based on the image and clinical history below, provide: DIAGNOSIS and FINDINGS"）是合适的，但应避免要求step-by-step reasoning。

### 3. 保留temperature=0（优先级：高）
官方文档和benchmark评估均使用temperature=0。当前配置正确。

### 4. 系统消息可选（优先级：低）
我们已验证无系统消息效果更好。这与研究结论一致（系统消息对domain-specific模型影响不显著）。

### 5. max_tokens=2048偏高（优先级：中）
官方推荐图像任务max_tokens=200-300。设置2048可能导致：
- 不必要的token消耗（增加成本和延迟）
- 可能触发MedGemma 4B的某些生成问题
建议降至500-1000，保留足够空间但不过度。

### 6. 确认模型版本包含v1.0.1修复
确保Vertex AI Model Garden部署的是2025年7月9日之后的模型版本，以确保end-of-image token bug已修复。

---

## Open Questions

1. Vertex AI Model Garden目前部署的MedGemma 4B是v1.0.0还是v1.0.1（包含end-of-image token修复）？

2. 当图像作为base64数据URI传递时，Vertex AI定制vLLM是否完全支持？（官方notebook只展示URL方式）

3. 对于MedPix这样的图像分类任务，使用专家角色系统消息（"You are an expert radiologist"）是否比无系统消息更好？

---

## Research Methodology Notes

- 总研究轮次：4轮（每个topic 3+轮）
- 使用工具：DeepWiki MCP, WebSearch, WebFetch
- 主要信息来源：Google-Health/medgemma官方repo（DeepWiki），官方HuggingFace模型页面，MedGemma Technical Report (arXiv:2507.05201)，Google Cloud文档
- 置信度：**高**（核心API格式和配置来自官方notebook代码）；**中**（prompt工程最佳实践来自第三方研究论文）；**低**（base64支持状态和确切模型版本未经直接验证）

---

## Source References

- Google-Health/medgemma官方repo: https://github.com/Google-Health/medgemma
- Model Garden Notebook: https://github.com/google-health/medgemma/blob/main/notebooks/quick_start_with_model_garden.ipynb
- MedGemma HuggingFace (4B): https://huggingface.co/google/medgemma-4b-it
- MedGemma Technical Report: https://arxiv.org/html/2507.05201v3
- Google Health AI Developer Docs: https://developers.google.com/health-ai-developer-foundations/medgemma
- MedGemma Model Card: https://developers.google.com/health-ai-developer-foundations/medgemma/model-card
- Vertex AI vLLM serving: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/vllm/use-vllm
- MedSigLIP HuggingFace: https://huggingface.co/google/medsiglip-448
- vLLM base64 bug issue: https://github.com/vllm-project/vllm/issues/28123
- MedGemma bug fix forum: https://discuss.ai.google.dev/t/bug-fix-in-medgemma-4b-it/94470
