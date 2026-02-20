# PRESCREEN 模块评估报告 v2.0
> **日期：** 2026-02-20
> **状态：** 最终稿 — 已整合多智能体团队报告
> **作者：** Claude 主团队 + gap-analyst + api-researcher (三路并行验证)
> **目的：** 对现有 PRD §7.2 和 SoT Harness §4.2 的 PRESCREEN 设计进行深度批判性评估，提出改进后的完整设计方案

---

## 0. 执行摘要

通过直接 CT.gov API v2 验证实验（使用示例患者 NSCLC + EGFR L858R）以及对 PRD v3.1 和 SoT Harness 的系统性审查，发现了 **6 项 CRITICAL 级漏洞**，其中最严重的是：

> **SoT Harness 设计中的"首要查询策略"（Q1：biomarker-first）实际上完全无法找到 anchor trial NCT05456256。**
> 对该试验而言，正确的查询策略是基于临床表型（`never smoker` + `TKI progression`），而非 EGFR L858R 生物标记。

这一发现根本性地改变了 PRESCREEN 模块的设计方向：从 **"biomarker-first"** 转向 **"clinical-phenotype-first + biomarker-specific 多层查询架构"**。

---

## 1. 直接 API 验证实验结果

### 1.1 实验场景

**患者档案（Scenario 1 Eligible）：** Stage IV NSCLC 腺癌，EGFR L858R exon 21，从不吸烟，ECOG 1，一线奥希替尼 14 个月后进展，58 岁女性。

**锚点试验：** NCT05456256 — HARMONIC Study（LP-300 + 卡铂 + 培美曲塞，针对 TKI 后进展的从不吸烟者）。

### 1.2 五种查询策略对比

| # | 查询策略 | API 参数 | NCT05456256 命中 | 结果总数 |
|---|--------|---------|:---:|:---:|
| **Q1** | 条件 + EGFR L858R 生物标记 | `condition=NSCLC, eligibility_keywords=EGFR L858R` | **❌ 未命中** | 20+ |
| **Q2** | 条件 + 干预（osimertinib）| `condition=NSCLC, intervention=osimertinib, phase=[P2,P3]` | **❌ 未命中** | 15+ |
| **Q3** | 宽泛 EGFR 激活突变 | `eligibility_keywords=actionable genomic alteration TKI` | **❌ 未命中** | **0 结果** |
| **Q4** | 临床表型（never smoker）| `eligibility_keywords=never smoker TKI progression` | **✅ 唯一命中** | 1 |
| **Q5** | Essie 语法精确匹配 | `AREA[EligibilityCriteria]"never smoker" + condition=lung adenocarcinoma + phase=PHASE2` | **✅ 唯一命中** | 1 |

### 1.3 根因分析

**为什么 Q1-Q3 失败？**

NCT05456256 的实际纳入标准原文使用的是：
> _"specific actionable genomic alterations (e.g., mesenchymal epithelial transition (MET) exon14 skipping mutations, anaplastic lymphoma kinase (ALK), epidermal growth factor receptor (EGFR), neurotrophic tyrosine receptor kinase (NTRK) fusions, etc.)"_

该文本包含 `EGFR`（通用词），但**不包含** `L858R` 或 `EGFR L858R`。CT.gov API 的 eligibility_keywords 搜索是基于索引词匹配，而非语义等价。

此外，`"actionable genomic alteration TKI"` 这个完整短语完全没有被索引，返回 0 结果 —— 证明长复合医学短语在 CT.gov 索引中不可靠。

**为什么 Q4/Q5 成功？**

`never smoker` 是该试验最独特的临床表型特征，直接出现在标题和资格标准中。这类**临床表型词**（非生物标记词）在 CT.gov 中的索引比复杂生物标记更可靠。

---

## 2. 现有设计批判性分析

### 2.1 CRITICAL 级漏洞

| # | 漏洞 | 影响范围 | 根本原因 |
|---|-----|---------|--------|
| **C-1** | SoT Harness Q1 查询策略（biomarker-first）无法找到 anchor trial | PRESCREEN 功能验证失败 | CT.gov 索引机制误解：trials 使用宽泛生物标记语言，非具体突变名 |
| **C-2** | `generate_search_terms()` 函数签名和返回类型完全未定义 | 模块无法实现 | PRD 只描述了功能，未指定接口契约 |
| **C-3** | SearchAnchors Pydantic 数据模型未定义 | 阻塞 INGEST→PRESCREEN 数据流 | prescreen/__init__.py 为空 |
| **C-4** | CT.gov API client 未定义（封装方式、错误处理、重试策略均缺失）| 任何 API 查询均无法执行 | PRD 只说"调用 API"但无实现规范 |
| **C-5** | eGFR vs EGFR 问题在 Essie 语法查询中仍存在 | 肾功能检查假阳性污染 EGFR 突变搜索 | SoT harness 提到但未给出完整解决方案 |
| **C-6** | SoT Harness 中 Scenario 2（EXCLUDED）的排除原因与实际试验标准不符 | 测试通过但临床推理错误 | 对 NCT05456256 排除标准 #4 的误读 |

#### C-6 详解：Scenario 2 EXCLUDED 的错误

SoT Harness 文档声称 Scenario 2 因"prior chemotherapy progression"触发 Exclusion criterion #4。

**但实际排除标准 #4 原文是：**
> _"Patients who have received chemotherapy and/or immunotherapy **but transitioned to a TKI with no evidence of disease progression** will be allowed to enroll. Patients who experienced **disease progression while on chemotherapy and/or immunotherapy** will be ineligible."_

**关键区别：**
- 化疗后无进展→转 TKI：**允许入组**
- 化疗期间发生进展：**排除**

SoT Harness 的 Scenario 2 需要明确说明患者是在化疗"期间"进展（不是化疗后进 TKI），否则临床推理错误会传播到 VALIDATE 评估。

---

### 2.2 HIGH 级漏洞

| # | 漏洞 | 影响范围 |
|---|-----|---------|
| **H-1** | 多查询结果合并/去重策略未定义 | 同一 NCT ID 从多个查询返回时无处理逻辑 |
| **H-2** | 分页策略缺失（CT.gov 默认 pageSize=10，PRESCREEN 需 50+）| 只取第 1 页结果，漏掉后续相关试验 |
| **H-3** | Phase 过滤不完整：PHASE1/PHASE2 联合期试验被排除 | NSCLC 领域大量早期新药试验为 Phase 1/2，固定过滤 PHASE2/3 会遗漏 |
| **H-4** | Prior therapy → negative anchors 的逻辑转化未定义 | osimertinib 先用→negative anchor "first-line treatment" 的推理规则未编码 |
| **H-5** | 缓存键策略（ingest_source=gold/model）未延伸到 CT.gov API 调用层 | 不同 ingest 来源触发相同 API 调用，run 间结果可能不一致 |
| **H-6** | CT.gov API 时间漂移：live API 返回当前试验状态，非 2021 快照 | qrel_coverage@50 指标在当前 API 上失效（试验可能已关闭） |
| **H-7** | 速率限制处理未定义（40 req/min），多 topic 并发时会触发 429 | API 请求失败后无重试/backoff 策略 |

---

### 2.3 MEDIUM 级漏洞

| # | 漏洞 |
|---|-----|
| **M-1** | NSCLC 分子亚型查询策略不对称：EGFR/ALK/ROS1/MET/RET/KRAS 各有不同最优查询词 |
| **M-2** | `qrel_coverage@50` 指标计算方法未指定代码实现路径 |
| **M-3** | UMLS/RxNorm/HUGO 规范化：MUST-anchor recall 评估依赖规范化，但无具体实现 |
| **M-4** | 对比试验存在相互矛盾：NCT05456256 同时接受 ALK/MET/NTRK 患者，但 SoT 只展示 EGFR 场景 |
| **M-5** | SoT Harness 的 `05_api_search_response.json` 录制时间固化，随时间失效 |
| **M-6** | 无结构化 JSON 输出验证（prompt 要求输出 JSON，但无 schema 验证代码） |
| **M-7** | run_manager 集成未为 CT.gov API 调用定义 cost_tracking 格式 |

---

## 3. 改进后的 PRESCREEN 架构设计

### 3.1 核心设计原则（修订）

**原则 1：临床表型优先，生物标记补充**
CT.gov 中不同试验对相同分子特征使用不同级别的特异性描述（有的写"EGFR L858R"，有的写"EGFR activating mutation"，有的写"actionable genomic alteration"）。单一生物标记查询会系统性地漏掉使用宽泛语言的试验。正确策略是用临床表型（吸烟史、治疗线数、ECOG、先前治疗药物）作为补充策略。

**原则 2：多层查询 + 结果聚合去重**
最多 5 个平行查询，按精确度递减排序（Q1 最精确，Q5 最宽泛），结果按 NCT ID 去重，按命中查询数量排名。

**原则 3：CT.gov 索引现实主义**
Essie `AREA[EligibilityCriteria]` 短语搜索有效；复杂复合医学短语不可靠；`never smoker`/`treatment naive` 等临床表型词比复杂突变表达式更可靠。

**原则 4：组件隔离（继承 ADR-002）**
PRESCREEN 独立评估时接受 gold INGEST SoT 作为输入；cache key 包含 `ingest_source=gold|model`。

---

### 3.2 数据模型设计

```python
# src/trialmatch/prescreen/schema.py

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field

class AnchorPriority(str, Enum):
    MUST = "MUST"
    SHOULD = "SHOULD"
    NICE_TO_HAVE = "NICE_TO_HAVE"

class ConditionAnchor(BaseModel):
    term: str                        # e.g., "non-small cell lung cancer"
    priority: AnchorPriority
    synonyms: list[str] = Field(default_factory=list)  # e.g., ["NSCLC", "lung adenocarcinoma"]

class BiomarkerAnchor(BaseModel):
    term: str                        # e.g., "EGFR L858R"
    priority: AnchorPriority
    normalized_form: str             # HUGO gene + HGVS notation: "EGFR p.L858R"
    search_variants: list[str] = Field(default_factory=list)
    # ["EGFR L858R", "EGFR exon 21 L858R", "EGFR activating mutation"]
    # Ordered by specificity (most → least specific)
    notes: str | None = None

class InterventionAnchor(BaseModel):
    term: str                        # e.g., "osimertinib"
    priority: AnchorPriority
    role: str                        # "prior_therapy" | "target_therapy" | "class"
    notes: str | None = None

class ClinicalPhenotypeAnchor(BaseModel):
    """NEW: Clinical phenotype traits searchable in eligibility criteria text."""
    term: str                        # e.g., "never smoker"
    priority: AnchorPriority
    search_phrase: str               # exact phrase for AREA[EligibilityCriteria] search
    # e.g., "never smoker", "treatment naive", "TKI progression"
    category: str                    # "smoking_history" | "performance_status" | "treatment_history"

class TrialConstraints(BaseModel):
    age: str | None = None           # e.g., "58"
    sex: str | None = None           # "MALE" | "FEMALE" | "ALL"
    phase: list[str] = Field(default_factory=lambda: ["PHASE1", "PHASE2", "PHASE3"])
    # NOTE: Default includes PHASE1 (early-phase trials) — often excluded in prior design
    status: list[str] = Field(default_factory=lambda: ["RECRUITING", "NOT_YET_RECRUITING"])
    geo: str | None = None

class NegativeAnchor(BaseModel):
    term: str                        # e.g., "first-line treatment"
    reason: str                      # Rationale for exclusion
    search_phrase: str               # How to express in CT.gov query (if applicable)

class SearchAnchors(BaseModel):
    """PRESCREEN output — input to CT.gov API query builder."""
    topic_id: str
    ingest_source: str               # "gold" | "model_medgemma" | "model_gemini"
    conditions: list[ConditionAnchor]
    biomarkers: list[BiomarkerAnchor]
    interventions: list[InterventionAnchor]
    clinical_phenotypes: list[ClinicalPhenotypeAnchor]  # NEW dimension
    constraints: TrialConstraints
    negative_anchors: list[NegativeAnchor]
    search_strategy_notes: str | None = None

class TrialSearchResult(BaseModel):
    """Single trial from CT.gov API search."""
    nct_id: str
    title: str
    brief_title: str
    status: str
    phase: list[str]
    conditions: list[str]
    interventions: list[str]
    sponsor: str
    enrollment: int | None
    start_date: str | None
    primary_completion_date: str | None
    locations_count: int | None
    source_queries: list[str] = Field(default_factory=list)
    # Which query strategies found this trial — for ranking/dedup

class TrialSearchResultSet(BaseModel):
    """Aggregated, deduplicated results across all query strategies."""
    topic_id: str
    search_anchors: SearchAnchors
    results: list[TrialSearchResult]
    query_log: list[dict]  # Full query parameters + response metadata for tracing
    total_api_calls: int
    total_results_before_dedup: int
    dedup_strategy: str              # "nct_id_union"
    cost_estimate_usd: float | None  # CT.gov API is free, but log latency
```

---

### 3.3 多层查询策略架构

#### 为 NSCLC + EGFR L858R + 后 TKI 进展患者设计的查询层

```
Layer 1 (最精确): 临床表型 + 特定疾病亚型
  → Q1: condition="lung adenocarcinoma" + AREA[EligibilityCriteria]"never smoker"
        + AREA[EligibilityCriteria]"tyrosine kinase inhibitor" + phase=PHASE2
  → 预期: 返回 NCT05456256 ✅

Layer 2 (精确): 疾病 + 具体生物标记
  → Q2: condition="non-small cell lung cancer"
        + eligibility_keywords="EGFR L858R"
        + status=RECRUITING
  → 预期: 返回 EGFR L858R 特异性试验，但可能遗漏宽泛语言试验

Layer 3 (中等): 疾病 + 先前干预
  → Q3: condition="NSCLC"
        + intervention="osimertinib"
        + status=RECRUITING, phase=[PHASE2, PHASE3]
  → 预期: 返回 osimertinib 相关试验（耐药、维持、联合）

Layer 4 (宽泛): 疾病 + 生物标记类别
  → Q4: condition="non-small cell lung cancer"
        + eligibility_keywords="EGFR activating mutation OR EGFR mutation"
        + status=RECRUITING
  → 预期: 覆盖使用宽泛 EGFR 语言的试验

Layer 5 (最宽泛，仅 SHOULD 层): 纯疾病查询作为候选池
  → Q5: condition="non-small cell lung cancer adenocarcinoma"
        + status=RECRUITING, phase=[PHASE2, PHASE3]
        + advanced_query=AREA[Phase:size]2（避免 Phase 1 only）
  → 预期: 提供宽泛候选集，由 VALIDATE 过滤

结果聚合:
  → 按 NCT ID 去重 (union)
  → 每个 NCT ID 记录 source_queries[] 以追踪哪个策略找到它
  → 按 len(source_queries) 降序排序（被更多策略命中的排前面）
```

#### 分子亚型查询策略映射表

| 分子亚型 | 最有效查询层 | 关键查询词 | 预期 Recall |
|--------|-----------|---------|-----------|
| EGFR L858R / Del19 | L2 精确生物标记 | `"EGFR L858R"`, `"EGFR exon 21"`, `"EGFR Del19"` | 高（常被明确列出）|
| EGFR 宽泛激活突变 | L4 宽泛生物标记 | `"EGFR activating mutation"`, `"EGFR mutation"` | 中（术语多样）|
| KRAS G12C | L2 精确生物标记 | `"KRAS G12C"`, `"KRAS p.G12C"` | 高（已标准化）|
| ALK 重排 | L2 精确生物标记 | `"ALK rearrangement"`, `"ALK fusion"`, `"ALK positive"` | 中（术语多样）|
| MET exon 14 | L2 精确生物标记 | `"MET exon 14"`, `"MET exon14 skipping"` | 中 |
| 从不吸烟后 TKI 进展 | **L1 临床表型** | `"never smoker" + "tyrosine kinase inhibitor"` | 高（表型独特）|
| RET 融合 | L2 精确生物标记 | `"RET fusion"`, `"RET rearrangement"` | 中 |
| ROS1 重排 | L2 精确生物标记 | `"ROS1 rearrangement"`, `"ROS1 fusion"` | 中 |

---

### 3.4 eGFR vs EGFR 问题完整解决方案

```python
# EGFR 生物标记查询时的精确化处理

# ❌ 有歧义：返回 eGFR（肾功能）相关试验
bad_query = 'AREA[EligibilityCriteria]"EGFR"'

# ✅ 正确：加限定词消除歧义
good_queries = [
    'AREA[EligibilityCriteria]"EGFR mutation"',
    'AREA[EligibilityCriteria]"EGFR L858R"',
    'AREA[EligibilityCriteria]"EGFR exon 19"',
    'AREA[EligibilityCriteria]"EGFR-activating"',
    'AREA[EligibilityCriteria]"epidermal growth factor receptor mutation"',
]

# 规则：永远不单独搜索 "EGFR" 或 "eGFR"
# 必须带上 "mutation"/"L858R"/"exon"/"activating"/"positive" 等修饰词
# 同理：
# ❌ "ALK" alone → 也可能匹配其他含义
# ✅ "ALK rearrangement", "ALK positive", "ALK fusion"
```

---

### 3.5 `generate_search_terms()` 函数规范

```python
# src/trialmatch/prescreen/generator.py

async def generate_search_terms(
    patient_note: str,
    key_facts: dict,          # Gold INGEST SoT 或 model INGEST 输出
    ingest_source: str,       # "gold" | "model_medgemma" | "model_gemini"
    adapter: ModelAdapter,
    max_tokens: int = 2048,
    timeout_seconds: float = 120.0,
) -> SearchAnchors:
    """
    将患者档案转化为 CT.gov 可用的搜索锚点。

    Args:
        patient_note: 原始 EHR 文本
        key_facts: INGEST 模块输出的结构化关键事实字典
        ingest_source: 标记 INGEST 来源，用于 cache key 隔离
        adapter: MedGemma 或 Gemini 适配器

    Returns:
        SearchAnchors: 结构化搜索锚点，包含 4 个维度：
          - conditions (疾病/条件)
          - biomarkers (生物标记)
          - clinical_phenotypes (临床表型 — NEW)
          - interventions (干预/药物)
          - constraints (年龄、性别、分期、招募状态)
          - negative_anchors (需排除的搜索词)

    Cache key: topic_id + ingest_source + model_name + prompt_version
    """
```

---

### 3.6 CT.gov API Client 设计

```python
# src/trialmatch/prescreen/ctgov_client.py

class CTGovClient:
    """CT.gov API v2 client with retry, pagination, rate limiting."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    RATE_LIMIT_RPM = 40
    DEFAULT_PAGE_SIZE = 50
    MAX_RETRIES = 3

    async def search(
        self,
        condition: str | None = None,
        intervention: str | None = None,
        status: list[str] | None = None,
        phase: list[str] | None = None,
        advanced_query: str | None = None,  # Essie 语法
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int = 3,                  # 最多取前 3 页（150 结果）
    ) -> list[TrialSearchResult]:
        """执行 CT.gov API 查询，处理分页和速率限制。"""

    async def execute_multi_query(
        self,
        search_anchors: SearchAnchors,
        query_strategies: list[QueryStrategy],
    ) -> TrialSearchResultSet:
        """并发执行多个查询策略，合并去重结果。"""
```

---

### 3.7 改进后的 SoT Harness 设计

#### 纠正 Scenario 1 的 PRESCREEN 查询规范

**原设计问题：** Q1 是 biomarker-first，但实际上无法找到 NCT05456256。

**修正后的 `04_api_query_spec.json`：**

```json
{
  "topic_id": "sot_001",
  "query_strategy": "multi_layer_5query",
  "queries": [
    {
      "query_id": "q1_phenotype_primary",
      "layer": 1,
      "description": "临床表型：肺腺癌 + never smoker + TKI 进展（最精确）",
      "expected_to_find_anchor_trial": true,
      "ct_gov_api_v2_params": {
        "query.cond": "lung adenocarcinoma",
        "query.term": "AREA[EligibilityCriteria](\"never smoker\" AND \"tyrosine kinase inhibitor\")",
        "filter.overallStatus": "RECRUITING",
        "filter.phase": "PHASE2",
        "pageSize": 50
      }
    },
    {
      "query_id": "q2_biomarker_specific",
      "layer": 2,
      "description": "精确生物标记：NSCLC + EGFR L858R 在资格标准中",
      "expected_to_find_anchor_trial": false,
      "ct_gov_api_v2_params": {
        "query.cond": "non-small cell lung cancer",
        "query.term": "AREA[EligibilityCriteria](\"EGFR L858R\" OR \"EGFR exon 21 L858R\")",
        "filter.overallStatus": "RECRUITING",
        "pageSize": 50
      },
      "notes": "验证实验证明此查询不命中 NCT05456256，因其使用宽泛的 EGFR 语言"
    },
    {
      "query_id": "q3_prior_intervention",
      "layer": 3,
      "description": "干预：osimertinib 后进展试验",
      "expected_to_find_anchor_trial": false,
      "ct_gov_api_v2_params": {
        "query.cond": "non-small cell lung cancer",
        "query.intr": "osimertinib",
        "filter.overallStatus": "RECRUITING",
        "pageSize": 50
      }
    },
    {
      "query_id": "q4_biomarker_broad",
      "layer": 4,
      "description": "宽泛生物标记：EGFR 激活突变",
      "expected_to_find_anchor_trial": false,
      "ct_gov_api_v2_params": {
        "query.cond": "non-small cell lung cancer",
        "query.term": "AREA[EligibilityCriteria](\"EGFR activating mutation\" OR \"EGFR mutation\")",
        "filter.overallStatus": "RECRUITING",
        "pageSize": 50
      }
    },
    {
      "query_id": "q5_broad_fallback",
      "layer": 5,
      "description": "宽泛候选池",
      "expected_to_find_anchor_trial": true,
      "ct_gov_api_v2_params": {
        "query.cond": "adenocarcinoma lung",
        "filter.overallStatus": "RECRUITING",
        "filter.phase": "PHASE2,PHASE3",
        "pageSize": 50
      }
    }
  ],
  "expected_nct_ids_in_combined_results": ["NCT05456256"],
  "expected_finding_query": "q1_phenotype_primary",
  "dedup_strategy": "nct_id_union",
  "notes": {
    "critical_finding": "Q1（临床表型策略）是唯一能找到 NCT05456256 的查询。biomarker-first 策略（Q2）在此场景失效。",
    "egfr_egfr_disambiguation": "AREA[EligibilityCriteria] 中必须使用 EGFR 带修饰词（mutation/L858R/activating）",
    "phase_strategy": "默认包含 PHASE1/2 联合期试验（不要硬过滤只留 PHASE2/3）",
    "never_smoker_indexing": "CT.gov API 可有效索引 'never smoker' 短语"
  }
}
```

#### Scenario 2 EXCLUDED 的修正

**修正患者档案定义（确保临床推理正确）：**

```
Scenario 2 排除场景应明确描述为：
- 患者曾接受卡铂+培美曲塞化疗（一线）→ 疾病在化疗期间进展
- 随后接受奥希替尼 TKI 治疗
- 当前 EGFR L858R，从不吸烟

排除原因：触发 Exclusion #4（"在化疗/免疫治疗期间出现疾病进展"）
而非：曾接受过化疗本身（那是允许的）
```

---

## 4. 关于 PRESCREEN 评估指标的修订

### 4.1 新增：多层查询 Recall 矩阵

原有 `qrel_coverage@50` 指标过于聚焦于单次查询。建议新增：

| 指标 | 定义 | 目标 |
|-----|-----|-----|
| **L1-phenotype-recall** | Q1（表型层）找到的相关试验比例 | — |
| **L2-biomarker-recall** | Q2（生物标记层）找到的相关试验比例 | — |
| **Combined-recall@50** | 所有层合并后找到的相关试验比例（前 50）| ≥ 70% |
| **MUST-anchor-recall** | 原有指标 | 100% |
| **SHOULD-anchor-recall** | 原有指标 | ≥ 80% |
| **qrel_coverage@50** | 原有指标（保留）| — |
| **Layer-attribution** | 每个相关试验由哪个查询层找到 | 用于分析哪层贡献最大 |

### 4.2 Negative Anchor Compliance 的 API 实现现实性

**原设计假设：** CT.gov API 支持 NOT 操作符排除负向锚点。
**实际情况：** CT.gov Essie 语法支持 `NOT` 但效果不稳定，且只能过滤基于索引字段的结果，不能对完整文本做 NOT 匹配。

**修正策略：**
- 负向锚点通过 VALIDATE 模块处理（让 VALIDATE 识别不适合的试验），而非在 PRESCREEN 层硬过滤
- PRESCREEN 只记录 negative_anchors，由后置过滤步骤处理

---

## 5. 完整的 NSCLC 患者 PRESCREEN 示例（更新版）

**患者：** 58 岁女性，Stage IV 肺腺癌，EGFR L858R，从不吸烟，ECOG 1，奥希替尼一线治疗 14 个月后进展

### 5.1 LLM Prompt 模板（修订版）

```
System:
Generate clinical trial search anchors for the given NSCLC patient.
Output structured JSON with 5 dimensions: conditions, biomarkers,
clinical_phenotypes, interventions, constraints, and negative_anchors.

IMPORTANT RULES:
1. clinical_phenotypes: Extract unique traits that appear as TEXT in eligibility
   criteria (e.g., "never smoker", "treatment naive", "TKI progression").
   These are DISTINCT from biomarkers — they describe clinical history,
   demographics, or lifestyle traits that trials use as inclusion/exclusion text.
2. biomarker search_variants: Include 3-4 variants from most→least specific.
   E.g., ["EGFR L858R", "EGFR exon 21", "EGFR activating mutation", "EGFR mutation"]
   NEVER use "EGFR" alone — disambiguate from eGFR (renal function).
3. interventions role: Mark prior therapies as "prior_therapy", targets as "target_therapy"
4. constraints.phase: Default ["PHASE1", "PHASE2", "PHASE3"] — don't restrict to PHASE2/3 only
5. negative_anchors: Explain why in "reason", give CT.gov-searchable phrase in "search_phrase"

User:
Patient profile: <<PROFILE_TEXT>>
Key facts: <<KEY_FACTS_JSON>>

Output JSON matching SearchAnchors schema:
{
  "topic_id": "...",
  "ingest_source": "gold|model",
  "conditions": [...],
  "biomarkers": [{"term": "...", "priority": "MUST|SHOULD", "normalized_form": "...",
                  "search_variants": ["most specific", "...", "least specific"]}],
  "clinical_phenotypes": [{"term": "...", "priority": "MUST|SHOULD",
                           "search_phrase": "exact phrase for CT.gov", "category": "..."}],
  "interventions": [...],
  "constraints": {"age": "...", "sex": "...", "phase": [...], "status": [...]},
  "negative_anchors": [{"term": "...", "reason": "...", "search_phrase": "..."}]
}
```

### 5.2 Gold SearchAnchors（修订版）

```json
{
  "topic_id": "sot_001",
  "ingest_source": "gold",
  "conditions": [
    {
      "term": "non-small cell lung cancer",
      "priority": "MUST",
      "synonyms": ["NSCLC", "lung adenocarcinoma", "adenocarcinoma of lung"]
    }
  ],
  "biomarkers": [
    {
      "term": "EGFR L858R",
      "priority": "MUST",
      "normalized_form": "EGFR p.L858R",
      "search_variants": [
        "EGFR L858R",
        "EGFR exon 21 L858R",
        "EGFR activating mutation",
        "EGFR mutation"
      ],
      "notes": "多变体搜索覆盖不同特异性级别；避免单独使用 EGFR（与 eGFR 歧义）"
    }
  ],
  "clinical_phenotypes": [
    {
      "term": "never smoker",
      "priority": "MUST",
      "search_phrase": "never smoker",
      "category": "smoking_history",
      "notes": "验证实验证明此为找到 NCT05456256 的关键查询词"
    },
    {
      "term": "TKI progression",
      "priority": "SHOULD",
      "search_phrase": "tyrosine kinase inhibitor",
      "category": "treatment_history",
      "notes": "与 never smoker 联合使用精确定位后 TKI 进展试验"
    }
  ],
  "interventions": [
    {
      "term": "osimertinib",
      "priority": "SHOULD",
      "role": "prior_therapy",
      "notes": "一线用药后进展——搜索后 osimertinib 进展选项"
    }
  ],
  "constraints": {
    "age": "58",
    "sex": "FEMALE",
    "phase": ["PHASE1", "PHASE2", "PHASE3"],
    "status": ["RECRUITING", "NOT_YET_RECRUITING"]
  },
  "negative_anchors": [
    {
      "term": "first-line treatment",
      "reason": "患者已接受过一线 TKI，一线试验的纳入标准通常排除此类患者",
      "search_phrase": "first-line",
      "note": "通过 VALIDATE 过滤，非 PRESCREEN API 查询过滤"
    },
    {
      "term": "treatment naive",
      "reason": "患者非初治——treatment naive 试验会排除",
      "search_phrase": "treatment naive",
      "note": "通过 VALIDATE 过滤"
    }
  ]
}
```

---

## 6. 决策矩阵：与原设计的对比

| 设计决策 | 原设计 | 修订设计 | 变更原因 |
|---------|------|---------|--------|
| 主查询策略 | Biomarker-first (EGFR L858R) | Phenotype-first (never smoker + TKI) + biomarker 补充 | API 验证：biomarker-first 无法找到 NCT05456256 |
| 查询层数 | 3 个固定查询 | 5 层递减精确度查询 | 覆盖不同特异性级别的试验描述 |
| 临床表型维度 | 无 | **新增** ClinicalPhenotypeAnchor | 关键缺失维度 |
| Phase 过滤 | 默认仅 PHASE2/3 | 默认 PHASE1/2/3（含早期联合期） | NSCLC 大量新药在 Phase 1/2 |
| Negative anchors 执行 | API NOT 查询 | 仅记录，由 VALIDATE 过滤 | CT.gov NOT 操作不稳定 |
| EGFR 搜索变体 | 单一 "EGFR L858R" | 4 变体从精确到宽泛 | 覆盖不同表达方式的试验 |
| Scenario 2 排除定义 | "prior chemotherapy" 模糊 | "化疗**期间**进展" 精确定义 | 对 NCT05456256 Exclusion #4 的正确解读 |
| 结果合并 | 未定义 | NCT ID 去重 + 多查询命中排名 | 必要功能，原缺失 |

---

## 7. 建议的实现路径

### Phase 0（当前冲刺）
1. 定义 `SearchAnchors` 和相关 Pydantic 模型（`prescreen/schema.py`）
2. 实现 `generate_search_terms()` 函数（Prompt + 解析 + 结构化输出）
3. 实现 `CTGovClient`（API 封装 + 分页 + 重试 + 速率限制）
4. 实现 `build_multi_layer_queries()`（SearchAnchors → 5 层查询参数）
5. 实现结果聚合去重（`aggregate_results()`）
6. 更新 SoT Harness 中的 `04_api_query_spec.json`（修正 Q1 为表型策略）
7. 修正 Scenario 2 患者档案定义（化疗期间进展 vs 化疗后转 TKI）
8. 编写单元测试（query builder、结果聚合、JSON 解析）

### Phase 1（后续）
1. 实现 `qrel_coverage@50` 指标计算
2. 实现 MUST-anchor recall 评估（UMLS 规范化）
3. 实现 Layer-attribution 分析
4. 实现完整的 PRESCREEN 孤立评估 CLI 命令
5. 添加 BDD 场景（MUST-anchor recall、multi-query dedup）

---

## 8. 遗留开放问题

| 问题 | 影响 | 优先级 |
|-----|-----|------|
| CT.gov API v2 的 Essie syntax 中 `filter.phase` 的精确值格式（`PHASE2` 还是 `Phase 2`？）| 查询参数正确性 | HIGH |
| 多查询并发执行时的速率限制策略（5 查询/topic × N topics 并发）| 吞吐量 | HIGH |
| UMLS CUI 规范化库选型（QuickUMLS？scispaCy？）| MUST-anchor recall 评估 | MEDIUM |
| never-smoker 表型是否在 TrialGPT HF 数据集的患者描述中存在？（验证 PRESCREEN 在 Phase 0 数据上的适用性）| Phase 0 数据兼容性 | HIGH |
| TrialGPT HF 数据集是否包含 CT.gov 试验 NCT IDs？（用于 extrinsic 评估）| Phase 0 PRESCREEN 评估可行性 | HIGH |

---

---

## 9. Agent 团队补充发现（整合自 gap-analyst + api-researcher）

### 9.1 CRITICAL 补充漏洞（来自 gap-analyst）

以下是 gap-analyst 发现的、本报告原版未覆盖的额外 CRITICAL 漏洞：

| # | 漏洞 | 影响 |
|---|-----|-----|
| **C-7** | MUST-anchor recall 评估依赖 UMLS/RxNorm/HUGO 规范化，但无具体实现规范 | 指标无法自动计算 |
| **C-8** | NSCLC 分子亚型搜索策略不对称：KRAS G12C 搜索 20+ 结果，ALK 搜索仅 3 结果——不同亚型需要不同策略 | 不同亚型的 Recall 极度不一致 |

### 9.2 Essie 语法关键发现（来自 api-researcher）

**重要区分**：

| Essie 语法变体 | 是否有效 |
|-------------|:---:|
| `AREA[EligibilityCriteria]"never smoker"` | **✅ 有效**（本报告验证） |
| `AREA[EligibilityModule.EligibilityCriteria]"..."` | **❌ 无效**（api-researcher 验证，报错"Unknown area name"）|

→ **结论：** 正确字段名是 `EligibilityCriteria`，不是 `EligibilityModule.EligibilityCriteria`。SoT Harness 中使用的语法（`AREA[EligibilityCriteria]`）是正确的。

### 9.3 更简单的等效查询方式（api-researcher 发现）

**发现：** 将 `never smoker` 直接并入 `condition` 参数比使用 Essie 语法更简洁且同样有效：

```python
# 等效查询，更简单：
search_trials(
    condition="non-small cell lung cancer never smoker",
    status=["RECRUITING"]
)
# → 返回 NCT05456256 排名第 1！

# vs 复杂的 Essie 方式：
search_trials(
    condition="lung adenocarcinoma",
    advanced_query='AREA[EligibilityCriteria]"never smoker"',
    phase="PHASE2"
)
```

### 9.4 分子亚型查询基准（api-researcher 实测）

| 亚型 | 查询参数 | 结果数量 | 回收质量 |
|-----|---------|:---:|:---:|
| KRAS G12C | `eligibility_keywords="KRAS G12C"` | 20+ | **极高**（sotorasib/adagrasib 等全覆盖）|
| EGFR L858R | `eligibility_keywords="EGFR L858R mutation"` | 4 | 高精度但**低 recall** |
| ALK 重排 | `eligibility_keywords="ALK rearrangement TKI"` | 3 | 稀少但精确 |
| MET exon 14 | `eligibility_keywords="MET exon 14 skipping"` | 9 | 合理（罕见靶点）|
| Never smoker | `eligibility_keywords="never smoker"` | 18 | 中等（noise 较高）|

**核心洞察：** KRAS G12C 的搜索效果远好于 EGFR（因为 KRAS G12C 已成为标准化命名，各试验均写一致），而 EGFR 因表达多样（L858R/exon 21/activating mutation/actionable alteration）导致任一查询都有 recall 损失。这进一步验证了**多变体 + 多层查询策略**的必要性。

### 9.5 NSCLC Phase 2/3 端点模式

根据 api-researcher 对 100 个 NSCLC Phase 2/3 试验的端点分析：

| 端点类型 | 主要终点 | 次要终点 |
|--------|:---:|:---:|
| ORR | 23% | 5% |
| Safety/DLT | 24% | 14% |
| PFS | 15% | 7% |
| OS | 5% | 11% |

→ NCT05456256 使用 PFS+OS 双主要终点的设计符合 Phase 2 常见模式。

---

## 附录：NCT05456256 完整资格标准关键点

> （从直接 API 调用获取，2026-02-20）

**纳入标准关键点（与 SoT 相关）：**
- #1: "specific actionable genomic alterations (**e.g., MET exon14, ALK, EGFR, NTRK**)" — 宽泛，不只是 EGFR L858R
- #3: "never smoker" — 可例外：有可操作突变的前吸烟者在特定情况可入组
- #4: TKI 后进展（注意：进展必须是 TKI 失败，不是化疗失败再转 TKI）
- #6: ECOG 0 或 1
- #10: ANC ≥ **1.5** x10^9/L（注意：SoT Gold 写的是 2.1，是在范围内，但临界值是 1.5）

**SoT Harness Scenario 1 Gold 数据核实：**
患者 ANC 2.1（✅ ≥ 1.5），Hgb 11.2（✅ ≥ 10），Plt 180（✅ ≥ 100），Cr 0.9（✅ ≤ 1.5mg/dL）— 所有实验室值通过

**排除标准 #4 完整解读（关键临床逻辑）：**
> 接受化疗/免疫治疗后转 TKI 且无进展的患者：**允许**
> 在化疗/免疫治疗期间发生疾病进展的患者：**排除**
> Scenario 2 必须使用后者定义
