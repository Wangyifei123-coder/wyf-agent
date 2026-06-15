"""Tests for tool permission-based access control"""

import pytest
from src.tools.registry import ToolRegistry, ToolSchema, ToolParameter, Tool


class MockTool(Tool):
    def __init__(self, name: str, allowed_roles: list[str] | None = None):
        self._name = name
        self._allowed_roles = allowed_roles if allowed_roles is not None else ["admin", "user"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=f"Mock tool {self._name}",
            parameters=[],
            allowed_roles=self._allowed_roles,
        )

    async def execute(self, **kwargs) -> str:
        return f"result_{self._name}"


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(MockTool("public_tool", allowed_roles=["admin", "user"]))
    reg.register(MockTool("admin_tool", allowed_roles=["admin"]))
    reg.register(MockTool("open_tool", allowed_roles=[]))
    return reg


def test_check_permission_admin_access(registry):
    """Admin should access all tools"""
    assert registry.check_permission("public_tool", "admin") is True
    assert registry.check_permission("admin_tool", "admin") is True
    assert registry.check_permission("open_tool", "admin") is True


def test_check_permission_user_access(registry):
    """User should access public and open tools, but not admin-only tools"""
    assert registry.check_permission("public_tool", "user") is True
    assert registry.check_permission("admin_tool", "user") is False
    assert registry.check_permission("open_tool", "user") is True


def test_check_permission_no_restrictions(registry):
    """Tools with empty allowed_roles should be accessible by anyone"""
    assert registry.check_permission("open_tool", "guest") is True
    assert registry.check_permission("open_tool", "anonymous") is True


def test_check_permission_nonexistent_tool(registry):
    """Non-existent tools should return False"""
    assert registry.check_permission("nonexistent", "admin") is False


@pytest.mark.asyncio
async def test_call_with_permission_allowed(registry):
    """Should execute tool when permission is granted"""
    result = await registry.call("public_tool", {}, user_role="user")
    assert result == "result_public_tool"


@pytest.mark.asyncio
async def test_call_with_permission_denied(registry):
    """Should return error when permission is denied"""
    result = await registry.call("admin_tool", {}, user_role="user")
    assert "permission denied" in result.lower()


@pytest.mark.asyncio
async def test_call_admin_tool_with_admin_role(registry):
    """Admin should be able to call admin-only tools"""
    result = await registry.call("admin_tool", {}, user_role="admin")
    assert result == "result_admin_tool"


def test_list_tools_for_role(registry):
    """Should return only tools accessible by the given role"""
    user_tools = registry.list_tools_for_role("user")
    user_tool_names = [t.name for t in user_tools]
    assert "public_tool" in user_tool_names
    assert "admin_tool" not in user_tool_names
    assert "open_tool" in user_tool_names

    admin_tools = registry.list_tools_for_role("admin")
    admin_tool_names = [t.name for t in admin_tools]
    assert "public_tool" in admin_tool_names
    assert "admin_tool" in admin_tool_names
    assert "open_tool" in admin_tool_names
