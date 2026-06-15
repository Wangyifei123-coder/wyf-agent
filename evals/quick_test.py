"""快速回归测试 - 只测试几个用例"""

import asyncio
import httpx

BASE_URL = "http://localhost:8081"

async def test():
    async with httpx.AsyncClient(timeout=120) as client:
        # 登录
        resp = await client.post(
            f"{BASE_URL}/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = resp.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 测试用例
        test_cases = [
            {"input": "Python有什么特点？", "keywords": ["简洁易读", "动态类型", "跨平台"]},
            {"input": "RAG技术是什么？", "keywords": ["检索增强生成", "检索", "生成"]},
            {"input": "你好", "keywords": ["你好", "hi", "hello"]},
        ]
        
        passed = 0
        for i, case in enumerate(test_cases, 1):
            resp = await client.post(
                f"{BASE_URL}/chat",
                json={"message": case["input"]},
                headers=headers,
            )
            data = resp.json()
            answer = data.get("answer", "")
            
            hits = sum(1 for kw in case["keywords"] if kw in answer)
            score = hits / len(case["keywords"])
            status = "PASS" if score >= 0.3 else "FAIL"
            
            if status == "PASS":
                passed += 1
            
            print(f"[{i}/{len(test_cases)}] {case['input'][:20]}... {status} score={score:.2f}")
        
        print(f"\nResult: {passed}/{len(test_cases)} passed")

asyncio.run(test())
