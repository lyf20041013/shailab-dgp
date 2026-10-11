# SHAILAB 对话 SFT 数据契约 v0.1

## 1. 目的
定义子项目一 DGP 对话类合成数据的标准格式，确保：
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
| messages | list[dict] | 是 | 多轮对话，每项含 role 和 content |
| tools | list[dict] | 否 | OpenAI 格式工具定义，P0 阶段不启用 |

### 3.2 messages 内部结构
| 字段 | 类型 | 必填 | 取值 | 说明 |
|---|---|---|---|---|
| role | string | 是 | system / user / assistant | P0 阶段仅支持这三种 |
| content | string | 是 | 文本 | assistant 内容不能为空 |

### 3.3 平台扩展字段（必填，VERL 忽略但平台使用）
| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| version_id | string | 是 | 12 位全局唯一不可变哈希 ID |
| meta | dict | 是 | 模板 ID、场景、角色、生成模型 |
| quality | dict | 质检后追加 | PASS/REVIEW/REJECT、规则命中、分数 |

### 3.4 meta 内部结构
| 字段 | 类型 | 说明 |
|---|---|---|
| template_id | string | 来源模板 ID |
| scene | string | 对话场景 |
| roles | list[string] | 对话角色列表 |
| generator_model | string | 生成模型名称 |

### 3.5 quality 内部结构（质检后追加）
| 字段 | 类型 | 说明 |
|---|---|---|
| label | string | PASS / REVIEW / REJECT |
| rule_hits | list[string] | 命中的质检规则 ID |
| scores | dict | 各维度得分 |

## 4. 样例数据（JSONL 格式）
```json
{
  "version_id": "a1b2c3d4e5f6",
  "messages": [
    {"role": "system", "content": "你是一名跨境报关员，负责和货主沟通报关流程。"},
    {"role": "user", "content": "我有一批锂电池要出口到德国。"},
    {"role": "assistant", "content": "锂电池出口需要提供 UN38.3 认证，请问您准备好了吗？"},
    {"role": "user", "content": "已经准备好了。"},
    {"role": "assistant", "content": "好的，还需要 MSDS 文件，请一并提供。"}
  ],
  "tools": [],
  "meta": {
    "template_id": "tpl-dialogue-001",
    "scene": "跨境报关",
    "roles": ["报关员", "货主"],
    "generator_model": "Qwen3-8B-Instruct"
  },
  "quality": {
    "label": "PASS",
    "rule_hits": [],
    "scores": {"structure": 1.0, "privacy": 1.0}
  }
}
