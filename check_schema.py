"""
检查对话数据是否符合 VERL MultiTurnSFTDataset 的格式要求。
使用本地 Qwen2.5-0.5B-Instruct tokenizer 做验证。
"""

import torch
from transformers import AutoTokenizer

# ========== 1. 加载本地 tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(
    "/home/jovyan/work/models/Qwen2.5-0.5B-Instruct",
    trust_remote_code=True,
    local_files_only=True,
)

# ========== 2. 测试样例 ==========
samples = [
    # 样例1：普通多轮对话
    {
        "version_id": "a1b2c3d4e5f6",
        "messages": [
            {"role": "system", "content": "你是一名跨境报关员，负责和货主沟通报关流程。"},
            {"role": "user", "content": "我有一批锂电池要出口到德国。"},
            {"role": "assistant", "content": "锂电池出口需要提供 UN38.3 认证，请问您准备好了吗？"},
            {"role": "user", "content": "已经准备好了。"},
            {"role": "assistant", "content": "好的，还需要 MSDS 文件，请一并提供。"},
        ],
        "tools": [],
    },
    # 样例2：带工具调用
    {
        "version_id": "b2c3d4e5f6a1",
        "messages": [
            {"role": "system", "content": "你是物流助手，可以调用汇率查询工具。"},
            {"role": "user", "content": "帮我算一下 1000 美元等于多少人民币。"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "get_exchange_rate",
                            "arguments": '{"from": "USD", "to": "CNY"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_001", "content": '{"rate": 7.18}'},
            {"role": "assistant", "content": "按当前汇率，1000 美元约等于 7180 元人民币。"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_exchange_rate",
                    "description": "查询汇率",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                        },
                        "required": ["from", "to"],
                    },
                },
            }
        ],
    },
]


# ========== 3. 结构检查 ==========
def check_role_structure(sample):
    messages = sample["messages"]
    issues = []

    valid_roles = {"system", "user", "assistant", "tool"}
    for i, m in enumerate(messages):
        if m["role"] not in valid_roles:
            issues.append(f"第 {i} 条消息 role 非法: {m['role']}")

    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            content = m.get("content", "")
            tool_calls = m.get("tool_calls", None)
            if not content and not tool_calls:
                issues.append(f"第 {i} 条 assistant 内容为空且无 tool_calls，loss 会为 0")

    call_ids = set()
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                call_ids.add(tc["id"])
    for m in messages:
        if m["role"] == "tool":
            if m.get("tool_call_id") not in call_ids:
                issues.append(
                    f"tool 消息的 tool_call_id={m.get('tool_call_id')} 没有对应的 assistant tool_calls"
                )

    if messages[0]["role"] not in ("system", "user"):
        issues.append(f"首条消息 role={messages[0]['role']}，应为 system 或 user")
    if messages[-1]["role"] != "assistant":
        issues.append(f"末条消息 role={messages[-1]['role']}，应为 assistant")

    if issues:
        print("⚠️  结构问题：")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ 结构检查通过")
    return len(issues) == 0


# ========== 4. 模拟 VERL sanity check ==========
def simulate_verl_check(tokenizer, sample):
    messages = sample["messages"]
    tools = sample.get("tools", None) or None

    full_inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_ids = full_inputs["input_ids"].squeeze(0)

    system_msg = [m for m in messages if m["role"] == "system"]
    system_len = 0
    if system_msg:
        sys_inputs = tokenizer.apply_chat_template(
            [system_msg[0]], add_generation_prompt=False, tokenize=True, return_tensors="pt"
        )
        system_len = sys_inputs["input_ids"].shape[1]

    concat_ids = []
    for i, msg in enumerate(messages):
        inputs = tokenizer.apply_chat_template(
            [msg],
            tools=tools if i == 0 else None,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
        )
        ids = inputs["input_ids"].squeeze(0)
        if i != 0 and msg["role"] != "system":
            ids = ids[system_len:]
        concat_ids.append(ids)

    concat_ids = torch.cat(concat_ids, dim=0)

    print(f"\n{'='*60}")
    print(f"version_id: {sample['version_id']}")
    print(f"整体拼接长度: {full_ids.shape[0]}")
    print(f"逐轮拼接长度: {concat_ids.shape[0]}")

    if torch.equal(full_ids, concat_ids):
        print("✅ PASS：逐轮拼接 == 整体拼接，VERL 可正常加载")
    else:
        print("❌ FAIL：逐轮拼接 != 整体拼接，VERL 会抛 AssertionError")
        min_len = min(len(full_ids), len(concat_ids))
        for i in range(min_len):
            if full_ids[i] != concat_ids[i]:
                print(f"   第一个不一致位置: index={i}")
                print(f"   full:   {tokenizer.decode(full_ids[max(0,i-5):i+10])}")
                print(f"   concat: {tokenizer.decode(concat_ids[max(0,i-5):i+10])}")
                break
        if len(full_ids) != len(concat_ids):
            print(f"   长度不一致: full={len(full_ids)}, concat={len(concat_ids)}")
    return torch.equal(full_ids, concat_ids)


# ========== 5. 主流程 ==========
if __name__ == "__main__":
    for sample in samples:
        print(f"\n{'#'*60}")
        print(f"# 测试 {sample['version_id']}")
        print(f"{'#'*60}")
        check_role_structure(sample)
        simulate_verl_check(tokenizer, sample)
