"""简单 MCP 服务器 — 提供计算器和天气查询工具"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WYF Test Server", json_response=True)


@mcp.tool()
def calculator(expression: str) -> str:
    """计算数学表达式

    支持基本运算: +, -, *, /, **
    示例: "2 + 3 * 4", "2 ** 10", "sqrt(16)"
    """
    import math

    allowed_names = {"sqrt": math.sqrt, "pi": math.pi, "e": math.e}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@mcp.tool()
def get_weather(city: str) -> str:
    """获取城市天气信息（模拟）

    Args:
        city: 城市名称，如 "北京", "上海", "广州"
    """
    weather_data = {
        "北京": "晴天，25°C，湿度 40%",
        "上海": "多云，22°C，湿度 65%",
        "广州": "阵雨，28°C，湿度 80%",
        "深圳": "阴天，26°C，湿度 70%",
        "杭州": "晴转多云，24°C，湿度 55%",
    }
    return weather_data.get(city, f"{city} 的天气信息暂不可用")


@mcp.tool()
def text_count(text: str) -> str:
    """统计文本的字符数、单词数和行数

    Args:
        text: 要统计的文本
    """
    chars = len(text)
    words = len(text.split())
    lines = len(text.split("\n"))
    return f"字符数: {chars}, 单词数: {words}, 行数: {lines}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
