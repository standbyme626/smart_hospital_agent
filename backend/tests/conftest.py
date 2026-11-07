import os

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "postgresql://sha:sha_pwd@127.0.0.1:5999/sha")
os.environ.setdefault("OLLAMA_ENABLED", "false")
os.environ.setdefault("HOSPITAL_MCP_URL", "http://127.0.0.1:5999/mcp")
os.environ.setdefault("OPENCODE_API_KEY", "test-key")
os.environ.setdefault("SILICONFLOW_API_KEY", "")
