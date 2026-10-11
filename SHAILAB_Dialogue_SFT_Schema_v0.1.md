# SHAILAB 对话 SFT 数据契约 v0.2

## 1. 目的
定义子项目一 DGP 对话类合成数据的标准格式，确保：
- 与《SHAILAB-SE-SRS-2026-SUB01-DGP-V1.0》8.2 节字段对齐；
- 能被 VERL MultiTurnSFTDataset 正确加载；
- 能被平台记录版本血缘、质检标签、任务来源；
- 能被前后端、质检模块、训练模块共同理解。

## 2. 数据格式
- 最终落盘格式：**Parquet**（VERL 源码 `pd.read_parquet` 直接读取）
- 中间交换格式：JSONL（便于人工检查与调试）

## 3. 字段定义

### 3.1 VERL 基础字段（必填）
| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| conversations | list[dict] | 是 | 多轮对话，每项含 role 和 content |
| tools | list[dict] | 否 | OpenAI 格式工具定义，P0 阶段不启用 |

**重要说明**：VERL 源码 `MultiTurnSFTDataset` 默认读取字段名为 `messages`（`config.messages_key`，默认值 `"messages"`）。本文档采用 `conversations` 与老师需求文档 8.2 节对齐，但**在喂给 VERL 前必须做以下二选一处理**：
- 方案 A：在 VERL 训练 config 中设置 `messages_key: conversations`；
- 方案 B：在数据落盘前，将 `conversations` 字段重命名为 `messages`。
否则 VERL 会因找不到 `messages` 字段而报错。

### 3.2 conversations 内部结构
| 字段 | 类型 | 必填 | 取值 | 说明 |
|---|---|---|---|---|
| role | string | 是 | user / assistant | P0 阶段仅支持这两种；system 信息已并入模板或首条 user |
| content | string | 是 | 文本 | assistant 内容不能为空 |

**注意**：老师文档 8.2 节的示例中，`conversations` 从 `user` 开始，没有单独的 `system` 消息。system 内容由模板固化，在生成时已隐含在场景与角色配置中。

### 3.3 平台扩展字段（必填，VERL 忽略但平台使用）
| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| data_id | string | 是 | 样本级唯一 ID，格式 `conv-YYYYMMDD-NNNNN` |
| version_id | string | 是 | 12 位全局唯一不可变哈希 ID |
| meta | dict | 是 | 场景、角色、轮数、生成模型 |
| quality | dict | 质检后追加 | PASS/REVIEW/REJECT、规则命中、分数 |

### 3.4 meta 内部结构
| 字段 | 类型 | 说明 |
|---|---|---|
| scene | string | 对话场景 |
| roles | list[string] | 对话角色列表 |
| turns | int | 对话轮数 |
| template_id | string | 来源模板 ID（对老师文档的扩展） |
| generator_model | string | 生成模型名称（对老师文档的扩展） |

### 3.5 quality 内部结构（质检后追加）
| 字段 | 类型 | 说明 |
|---|---|---|
| label | string | PASS / REVIEW / REJECT |
| rule_hits | list[string] | 命中的质检规则 ID |
| scores | dict | 各维度得分 |

## 4. 样例数据（JSONL 格式）

```json
{
  "data_id": "conv-20260911-00892",
  "version_id": "4a8e910bd281",
  "meta": {
    "scene": "跨境集装箱多式联运异常协商",
    "roles": ["货代调度员", "船公司客服"],
    "turns": 6,
    "template_id": "tpl-dialogue-001",
    "generator_model": "Qwen3-8B-Instruct"
  },
  "conversations": [
    {
      "role": "user",
      "content": "集装箱号 COSU8892182 预计今天上午到港，但港区报修导致装卸延迟，请协助确认最迟还箱时间免堆期。"
    },
    {
      "role": "assistant",
      "content": "已为您检索该票单据。该箱原定免堆期截止至9月18日18:00。鉴于港口设备故障已出具不可抗力说明，系统已自动向码头申请顺延24小时免堆期，最新截止时间为9月19日18:00。"
    }
  ],
  "tools": [],
  "quality": {
    "label": "PASS",
    "rule_hits": [],
    "scores": {"structure": 1.0, "privacy": 1.0}
  }
}
