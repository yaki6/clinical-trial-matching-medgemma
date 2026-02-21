# Deep Research Report: TrialGPT Prompting Strategy

**Date**: 2026-02-20
**Researcher**: prompt-researcher (deep-research-agent)
**Repositories Analyzed**: ncbi-nlp/TrialGPT (GitHub)
**Total Research Rounds**: 3

---

## Executive Summary

TrialGPT-Matching 使用两次独立的 API 调用处理每个 patient-trial pair：一次处理所有 inclusion criteria，一次处理所有 exclusion criteria。每次调用中，所有同类 criteria 以编号列表形式批量传入，模型一次性输出所有 criteria 的 JSON verdict。

关键设计哲学：**TrialGPT 维持 inclusion/exclusion 的语义独立**，使用不同的标签集合（included/not included vs excluded/not excluded），而不是统一到 MET/NOT_MET 再做翻转。这一设计避免了语义混淆，也是其达到 87.3% accuracy 的核心 prompt 设计决策。

与我们当前实现的最大差异在于：TrialGPT 每次处理一个 trial 的所有同类 criteria（批量），而我们每次只处理单个 criterion（one-shot）。此外 TrialGPT 使用分离的标签空间而非统一的 MET/NOT_MET，并包含明确的 "Closed World Assumption"（未提及的事实视为不存在）。

---

## Research Objectives

1. TrialGPT criterion-level prediction prompt 的完整文本
2. inclusion vs exclusion 的处理逻辑差异
3. chain-of-thought 结构
4. output format 和标签定义
5. 与我们当前 prompt 的关键差异
6. trial-level aggregation 逻辑及其 label 语义含义

---

## Detailed Findings

### Opening Item 1: TrialGPT-Matching Prompt 完整内容

#### Round 1: Surface Exploration

**Questions Asked**: TrialGPT-Matching 的 prompt 结构是什么？如何处理 inclusion/exclusion？

**Key Discoveries**:
- 函数：`trialgpt_matching/TrialGPT.py` 中的 `get_matching_prompt(trial_info, inc_exc, patient)`
- 返回 `(system_prompt, user_prompt)` 两部分
- `inc_exc` 参数控制是处理 "inclusion" 还是 "exclusion"
- 两次独立调用：一次 inclusion，一次 exclusion

**Initial Gaps**: 需要原始文本，不仅是描述

#### Round 2: Deep Dive - 原始 Prompt 文本

通过直接 curl 获取源代码，得到完整 prompt 文本：

**System Prompt（Inclusion 版本）**:

```
You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the inclusion criteria of a clinical trial to determine the patient's eligibility at the criterion level.

The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.

You should check the inclusion criteria one-by-one, and output the following three elements for each criterion:
    Element 1. For each inclusion criterion, briefly generate your reasoning process: First, judge whether the criterion is not applicable (not very common), where the patient does not meet the premise of the criterion. Then, check if the patient note contains direct evidence. If so, judge whether the patient meets or does not meet the criterion. If there is no direct evidence, try to infer from existing evidence, and answer one question: If the criterion is true, is it possible that a good patient note will miss such information? If impossible, then you can assume that the criterion is not true. Otherwise, there is not enough information.
    Element 2. If there is relevant information, you must generate a list of relevant sentence IDs in the patient note. If there is no relevant information, you must annotate an empty list.
    Element 3. Classify the patient eligibility for this specific inclusion criterion: the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "included" denotes that the patient meets the inclusion criterion, while "not included" means the reverse.

You should output only a JSON dict exactly formatted as: dict{str(criterion_number): list[str(element_1_brief_reasoning), list[int(element_2_sentence_id)], str(element_3_eligibility_label)]}.
```

**System Prompt（Exclusion 版本）**，差异仅在于第 2 段和 Element 3：

```
...The factors that disqualify someone from participating are called exclusion criteria...

...Element 3. Classify the patient eligibility for this specific exclusion criterion: the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. ... "excluded" denotes that the patient meets the exclusion criterion and should be excluded in the trial, while "not excluded" means the reverse.
```

**User Prompt**:
```
Here is the patient note, each sentence is led by a sentence_id:
{patient_note_with_sentence_ids}

Here is the clinical trial:
Title: ...
Target diseases: ...
Interventions: ...
Summary: ...
Inclusion/Exclusion criteria:
 0. criterion 0 text
 1. criterion 1 text
 ...

Plain JSON output:
```

**Key Discoveries**:
- Batch 模式：一个 prompt 包含一个 trial 的所有同类 criteria（编号 0, 1, 2...）
- 患者 note 以 sentence_id 编号形式传入
- 要求 `temperature=0`（确定性输出）
- 标签空间完全分离：inclusion 用 included/not included，exclusion 用 excluded/not excluded

#### Round 3: Crystallization

**Closed World Assumption（关键设计）**：

Element 1 包含这段推理指令：
> "If the criterion is true, is it possible that a good patient note will miss such information? If impossible, then you can assume that the criterion is not true."

Element 3 进一步强化：
> "Try to use as less 'not enough information' as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient."

这是 **Closed World Assumption（封闭世界假设）**：未提及 = 不存在。这极大减少了 UNKNOWN/not enough information 的数量，提升分类置信度。

**Validated Assumptions**:
- 每次 API 调用处理一个 trial 的一类（inclusion 或 exclusion）所有 criteria
- inclusion 和 exclusion 使用不同的语义标签，不做统一映射
- 模型输出为 JSON dict，key 是 criterion 编号（字符串），value 是 3 元素列表

---

### Opening Item 2: TrialGPT Prompt 设计哲学

#### Round 1: 是否在 Prompt 中处理 inclusion/exclusion 逻辑？

**Answer**: 是。TrialGPT 在 prompt 中明确分离 inclusion/exclusion 语义，使用不同的标签集合，而非让模型做统一的事实判断再后处理翻转。

**设计哲学总结**：

1. **语义分离原则**：inclusion 问"患者是否满足进入条件"，exclusion 问"患者是否触发排除条件"。标签直接对应这两种语义，避免歧义。

2. **批量处理原则**：同类 criteria 一次处理，让模型有完整上下文，避免割裂地看每个 criterion。

3. **Closed World Assumption**：主动指导模型采用封闭世界假设，减少 UNKNOWN 率，提高分类覆盖率。

4. **Evidence-grounded**：要求模型引用 sentence_id，增强可解释性和可验证性。

---

### Opening Item 3: 与我们当前 Prompt 的对比分析

#### 我们的当前 Prompt（evaluator.py）

**INCLUSION_INSTRUCTIONS**：
```python
"For INCLUSION criteria: MET = patient meets this requirement. "
"NOT_MET = patient fails to meet this requirement."
```

**EXCLUSION_INSTRUCTIONS**：
```python
"For EXCLUSION criteria: MET = patient HAS this characteristic and WOULD BE EXCLUDED "
"from the trial. NOT_MET = patient does NOT have this characteristic and is NOT excluded "
"by this criterion."
```

**PROMPT_TEMPLATE** 核心结构：
```
You are a clinical trial eligibility assessment expert.

Criterion Type: {criterion_type}
{criterion_type_instructions}

Criterion: {criterion_text}
Patient Note: {patient_note}

Think step by step:
1. What does this criterion specifically require?
2. What does the patient note explicitly state about this?
3. Is the evidence clear enough to conclude MET or NOT_MET, or is information missing?

Respond ONLY with valid JSON:
{"verdict": "MET" | "NOT_MET" | "UNKNOWN", "reasoning": "...", "evidence_sentences": [0, 1, 2]}
```

#### 关键差异对照表

| 维度 | TrialGPT | 我们的实现 |
|------|----------|-----------|
| **标签空间** | inclusion: {included, not included, not applicable, not enough info} / exclusion: {excluded, not excluded, not applicable, not enough info} | 统一: {MET, NOT_MET, UNKNOWN} |
| **处理粒度** | 批量：一次调用处理一个 trial 所有同类 criteria | 单次：每个 criterion 独立调用 |
| **Closed World Assumption** | 明确指令："医学事实未提及即视为不存在" | 无此指令（默认 UNKNOWN） |
| **Chain-of-thought 结构** | 3 步：判断 applicability → 找直接证据 → 推断或标 NEI | 3 步：criterion 要求 → patient 陈述 → 确定性判断 |
| **不适用处理** | 有 "not applicable" 标签 | 无，归入 UNKNOWN |
| **exclusion 语义** | "excluded" = 患者触发排除条件（直接语义） | MET = 患者有该特征且被排除（需要元认知） |
| **Applicability 引导** | 第一步明确判断 "is this criterion applicable?" | 无此步骤 |
| **输出格式** | JSON dict，key 为 criterion 编号 | JSON object，单 criterion |

#### 最关键差异的深层分析

**1. 统一标签 vs 分离标签**

我们用 MET/NOT_MET 统一两类 criteria，需要在 prompt 里用文字解释 exclusion 的 MET 含义（患者有该特征且被排除）。这是 **meta-level 解释**，要求模型理解"我用 MET 来表示'被排除'"，增加了认知负担。

TrialGPT 使用直接语义：exclusion criterion 的 "excluded" 就是字面意思"患者被排除"。模型不需要额外映射。

**2. Closed World Assumption 缺失**

我们的 prompt 说 "UNKNOWN: Insufficient information — do NOT guess; use this only when evidence is truly absent"，这实际上鼓励了 UNKNOWN 的使用。

TrialGPT 反其道而行："Try to use as less 'not enough information' as possible"。临床试验 notes 通常记录阳性发现，未提及的病史/特征通常不存在——CWA 更符合医疗文档的书写惯例。

**3. Applicability 缺失**

我们没有 "not applicable" 概念。例如：一个关于怀孕史的 exclusion criterion 对男性患者不适用，我们会强制输出 NOT_MET 或 UNKNOWN，而 TrialGPT 用 "not applicable" 优雅处理。

---

### Opening Item 4: Trial-Level Aggregation 逻辑

#### 来源：trialgpt_ranking/rank_results.py

**get_matching_score() 算法**：

```python
# inclusion 计数
included, not_inc, na_inc, no_info_inc = 0, 0, 0, 0
# exclusion 计数
excluded, not_exc, na_exc, no_info_exc = 0, 0, 0, 0

# 分数计算
score = included / (included + not_inc + no_info_inc + eps)  # inclusion 满足比例
if not_inc > 0: score -= 1   # 任何 inclusion criterion NOT MET → 大惩罚
if excluded > 0: score -= 1  # 任何 exclusion criterion triggered → 大惩罚
```

**关键语义含义**：

- `not applicable` 不参与计分（分子分母都不计）
- `not enough information` 计入分母但不计入分子（等效于 not_inc 的"软版本"）
- `not_inc` = 1 就导致 score -= 1（一票否决 inclusion）
- `excluded` = 1 就导致 score -= 1（一票否决 exclusion）
- **na 不影响分数**：不适用的 criterion 完全透明

**对我们 label 语义的启示**：

TrialGPT 的 trial-level label 映射为二元：
- 最终 eligible = inclusion 全部 "included" + exclusion 全部 "not excluded"（排除 na 后）
- 对应我们的 TrialGPT HF 数据集中：expert_eligibility 的正面判断

在 TrialGPT HF 数据集中，6 个 criterion-level labels 对应：
- `included` (2) → 满足 inclusion criterion
- `not included` (1) → 不满足 inclusion criterion
- `excluded` (1) → 触发 exclusion criterion（患者应被排除）
- `not excluded` (2) → 未触发 exclusion criterion
- `not applicable` (3) → 该 criterion 不适用于此患者
- `not enough information` (0) → 信息不足

---

## Cross-Cutting Insights

1. **TrialGPT 的设计假设**：患者 note 是结构化的临床记录，写了什么就是什么，未写的视为不存在。这比通用对话 LLM 的"谨慎 UNKNOWN"更符合临床实践。

2. **批量处理的隐性优势**：模型一次看到所有 inclusion criteria，可能产生一致性更好的推理（如果 criterion 1 和 criterion 3 相互关联，批量模式能更好处理）。

3. **Label 语义自洽**：TrialGPT 的标签设计使得 inclusion 和 exclusion 都有"正面结果"（included/not excluded）和"负面结果"（not included/excluded），直接反映临床试验的招募逻辑。

---

## Architecture/Design Decisions

### TrialGPT 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 标签分离 | 两套标签（inclusion/exclusion各自） | 避免元认知负担，语义直接 |
| 批量处理 | 一次 call 处理所有同类 criteria | 减少 API 调用，保持上下文一致性 |
| CWA | 明确告知模型采用封闭世界假设 | 医疗文档惯例：未记录≈不存在 |
| not applicable | 独立标签，不参与分数 | 优雅处理性别/年龄等预设条件 |
| temperature=0 | 确定性输出 | 可重现性和评估一致性 |

---

## Edge Cases & Limitations

1. **批量处理的风险**：如果 trial 有大量 criteria，单次 prompt 可能超出上下文限制
2. **CWA 的局限**：对于诊断性信息（如"患者无糖尿病"），CWA 可能正确；但对于罕见病特征，未提及可能只是记录不完整
3. **not applicable 的判断**：这本身也需要推理，对于模糊情况（如"患者可能不适用"）仍有歧义

---

## Recommendations

基于对比分析，我们的 prompt 应考虑以下改进：

1. **引入 CWA 指令**：将 "do NOT guess" 改为 "if a medical fact is not mentioned in the note, assume it is not true for the patient (closed world assumption). Use UNKNOWN only when the criterion's premise itself cannot be assessed."

2. **考虑 not applicable 标签**：对于不适用的 criteria（如男性患者遇到妊娠相关 exclusion），明确引导模型输出 NOT_MET 而非 UNKNOWN，并在 label 映射时特殊处理。

3. **Exclusion 语义简化**：考虑将 exclusion prompt 改为直接问"Is the patient EXCLUDED by this criterion?"，而非解释 MET 的元语义。

4. **Element 1 applicability check**：在 CoT 第一步加入"Is this criterion applicable to this patient?"的明确判断。

---

## Open Questions

- 我们是否应该迁移到批量处理模式（一次处理一个 trial 的所有 criteria）？这会影响成本模型（更少 API 调用，但每次 call token 更多）。
- CWA 是否会降低 UNKNOWN 率而提升 MET/NOT_MET 的错误分类？需要 ablation 实验验证。

---

## Research Methodology Notes

- Round 1：通过 WebSearch 定位正确的仓库（ncbi-nlp/TrialGPT，不是 ncbi/TrialGPT）
- Round 2：通过 curl 直接获取原始源代码，获得完整 prompt 文本
- Round 3：通过 DeepWiki 验证批量处理模式，并研究 aggregation 逻辑
- **置信度**：HIGH — 基于原始源代码，非描述性文档

**关键源文件**：
- `trialgpt_matching/TrialGPT.py` — `get_matching_prompt()` 函数，行 ~58-100
- `trialgpt_ranking/rank_results.py` — `get_matching_score()` 函数，aggregation 算法
