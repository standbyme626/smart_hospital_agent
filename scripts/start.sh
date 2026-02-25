#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏥 正在启动智能分诊系统...${NC}"

# 0. 清理旧进程 (防止端口冲突)
echo "正在清理旧进程..."
pkill -f "uvicorn app.main:app" || true
pkill -f "streamlit run frontend/app.py" || true
sleep 1

# 1. 激活环境
source venv/bin/activate

# 2. 基础设施自检
echo -e "${BLUE}[1/4] 正在检查 RAG (Milvus/Redis) 连接...${NC}"
python check_rag_status.py
if [ $? -ne 0 ]; then
    echo -e "${GREEN}检查失败，服务启动终止。${NC}"
    exit 1
fi

# 3. 启动后端 (后台运行)
echo -e "${GREEN}[2/4] 启动后端 API (Port 8000)...${NC}"
cd backend
# 使用 nohup 但不重定向，或者重定向到文件但同时用 tail
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# 等待后端启动
echo "等待后端就绪 (5s)..."
sleep 5

# 检查后端是否存活
if ! ps -p $BACKEND_PID > /dev/null; then
    echo "❌ 后端启动失败! 请查看 backend.log:"
    cat backend.log
    exit 1
fi

# 4. 启动前端
echo -e "${GREEN}[3/4] 启动前端 UI (Port 8501)...${NC}"
nohup streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 > frontend.log 2>&1 &
FRONTEND_PID=$!

echo -e "${GREEN}[4/4] 服务已启动!${NC}"
echo -e "👉 前端访问地址: ${BLUE}http://localhost:8501${NC}"
echo -e "� 正在显示后端实时日志 (Ctrl+C 停止查看日志，服务仍在后台运行)..."
echo -e "💡 若要完全停止服务，请运行: kill $BACKEND_PID $FRONTEND_PID"

# 实时显示后端日志
tail -f backend.log
