import os
import json
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

# ================= 对应要求 1：环境变量 =================
load_dotenv() 
# 从环境变量读取，绝对不把 sk-xxx 写死在代码里
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com/v1" # 对应要求 2：OpenAI 兼容协议
)

# ================= 对应要求 3：3 个本地工具 =================
def run_bash(cmd: str) -> str:
    """执行终端命令"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return f"标准输出:\n{result.stdout}\n错误输出:\n{result.stderr}"

def read_file(path: str) -> str:
    """读取文件内容"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {e}"

def write_file(path: str, content: str) -> str:
    """写入文件"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功写入文件: {path}"
    except Exception as e:
        return f"写入失败: {e}"

# ================= 对应要求 3：工具的“说明书” =================
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "执行终端命令并返回输出",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "description": "要执行的命令"}},
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定路径的文件内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件路径"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入指定路径的文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"}
                },
                "required": ["path", "content"]
            }
        }
    }
]

# 把工具名字和实际函数对应起来
tool_map = {"run_bash": run_bash, "read_file": read_file, "write_file": write_file}

# ================= 对应要求 4 & 5：Agent Loop 与 Trace =================
def agent_loop(user_input: str, max_rounds: int = 10):
    messages = [
        {"role": "system", "content": "你是一个能使用工具的AI助手。请合理调用工具完成任务，完成后直接回复。"},
        {"role": "user", "content": user_input}
    ]

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*20} 第 {round_num} 轮对话开始 {'='*20}")
        
        # 1. 请求模型
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        # 如果没有工具调用，说明任务结束
        if not msg.tool_calls:
            print(f"✅ [Trace - 模型最终回复]: {msg.content}")
            break

        # 把模型的工具调用请求加到历史记录里
        messages.append(msg)

        # 2. 解析模型请求，本地执行
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            args = json.loads(tc.function.arguments)
            
            # 真正执行函数
            try:
                result = tool_map[tool_name](**args)
            except Exception as e:
                result = f"执行出错: {e}"

            # ================= 对应要求 5：打印 Trace =================
            print(f"🔧 [Trace - 工具调用]: 调用了 `{tool_name}`")
            print(f"📥 [Trace - 传入参数]: {args}")
            print(f"📤 [Trace - 返回结果]: {result.strip()[:200]}...") # 截断防止太长刷屏

            # 3. 结果回填消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })

if __name__ == "__main__":
    task = input("请输入你要AI完成的任务: ")
    agent_loop(task)