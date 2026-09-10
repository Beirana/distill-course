#!/usr/bin/env python3
"""双服务自由对话：连接已启动的 vllm serve（默认 8001），多轮交互。

用法（实例上）：
    python chat.py 8001 base          # 与蒸馏前（学生基座）对话
    python chat.py 8002 distilled     # 与蒸馏后（merged）对话

第二个参数填 `curl -s http://127.0.0.1:<port>/v1/models` 返回的 id
（起服务时用了 --served-model-name 就是别名，否则是模型全路径）。
退出：输入 exit / quit / 退出，或 Ctrl+C。
归档说明：录屏主片的定题对照用 ask_both.py（一次双列）；本脚本供自由演示与试手感。
"""
import json
import sys
import urllib.request

SYS = ("You are a helpful assistant. Solve the math problem step by step. "
       "End with a separate line: Answer: <number>. Use a signed integer "
       "or decimal without units on that line.")


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "8001"
    model = sys.argv[2] if len(sys.argv) > 2 else "base"
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    msgs = [{"role": "system", "content": SYS}]
    print(f"已连 {url}（model={model}）。输入问题可多轮追问；exit 退出。")
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit", "退出"):
            break
        msgs.append({"role": "user", "content": q})
        body = json.dumps({"model": model, "messages": msgs,
                           "temperature": 0, "top_p": 1.0, "repetition_penalty": 1.0,
                           "max_tokens": 768}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            ans = json.load(resp)["choices"][0]["message"]["content"]
        msgs.append({"role": "assistant", "content": ans})
        print(f"\n模型: {ans}")


if __name__ == "__main__":
    main()
