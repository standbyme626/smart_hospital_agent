import os
import sys
from unittest.mock import MagicMock
from dotenv import load_dotenv

# 1. 加载环境变量
# 禁用 CrewAI 遥测以避免交互式提示卡死
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

# 从项目根目录加载 .env 文件
# 文件位于 /home/kkk/Project/smart_hospital_agent/.env
# verify_crew.py 位于 backend/app/core/crew/verify_crew.py
# 需要向上回溯 4 层: crew -> core -> app -> backend -> smart_hospital_agent
# 也就是 ../../../../.env
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.abspath(os.path.join(current_dir, "../../../../.env"))
print(f"尝试从以下路径加载 .env: {env_path}")

if os.path.exists(env_path):
    print("文件存在。")
else:
    print("警告: .env 文件不存在！")

# 强制覆盖加载
loaded = load_dotenv(env_path, override=True)
print(f"load_dotenv 返回结果: {loaded}")

# 强制检查并在缺失时手动解析 (防止解析问题)
if 'OPENAI_API_KEY' not in os.environ:
    print("load_dotenv 后 OPENAI_API_KEY 仍缺失。尝试手动解析...")
    from dotenv import dotenv_values
    config = dotenv_values(env_path)
    if 'OPENAI_API_KEY' in config:
        os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
        print("已从 dotenv_values 手动设置 OPENAI_API_KEY")
    
    # 设置其他相关 Key
    if 'OPENAI_MODEL_NAME' in config:
        os.environ['OPENAI_MODEL_NAME'] = config['OPENAI_MODEL_NAME']
    if 'OPENAI_API_BASE' in config:
        os.environ['OPENAI_API_BASE'] = config['OPENAI_API_BASE']

print(f"OPENAI_API_KEY 设置状态: {'OPENAI_API_KEY' in os.environ}")

# 2. 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../.."))
backend_path = os.path.join(project_root, "smart_hospital_agent", "backend")

# 将 backend 添加到 sys.path 以便导入 app 模块
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

print(f"已添加 {backend_path} 到 sys.path")

# 3. Mock MedicalRetriever (模拟医疗检索器)
# 由于环境可能缺少 torch 等重依赖，我们必须 Mock 检索器以验证 CrewAI 逻辑。
# LLM 调用将是真实的。
mock_retriever_module = MagicMock()
class MockMedicalRetriever:
    """模拟的医疗检索器，用于测试环境"""
    def search(self, query, top_k=3):
        print(f"\n[MockDB] 正在检索: {query}")
        return [
            {"content": "标准方案: 对于严重偏头痛，考虑使用舒马曲坦 50mg。", "source": "2024临床指南"},
            {"content": "药物相互作用: 舒马曲坦不应与 MAOIs 同时使用。", "source": "药品数据库"},
            {"content": "诊断辅助: 畏光和恶心是偏头痛的强指征。", "source": "默克诊疗手册"}
        ]
mock_retriever_module.MedicalRetriever = MockMedicalRetriever
sys.modules['app.rag.retriever'] = mock_retriever_module
print("已 Mock MedicalRetriever (使用模拟类以避免重依赖)")

# 4. 导入 Crew
try:
    from app.core.crew.expert_crew import MedicalExpertCrew
    print("成功导入 MedicalExpertCrew")
except ImportError as e:
    print(f"导入 MedicalExpertCrew 失败: {e}")
    sys.exit(1)

# 5. 运行验证
def main():
    """主验证函数：初始化 Crew 并执行真实 LLM 调用测试"""
    print("正在初始化医疗专家组 (真实 LLM 模式)...")
    try:
        # 初始化 Crew
        expert_crew = MedicalExpertCrew()
        crew_instance = expert_crew.crew()
        
        # 中文测试用例
        inputs = {
            'symptoms': '剧烈单侧头痛，搏动性，伴有畏光、恶心、呕吐。持续时间：6小时。',
            'medical_history': '女性，30岁。有发作性偏头痛病史。无已知过敏史。目前正在服用口服避孕药。'
        }
        
        print("\n[测试] 使用以下输入启动 Crew 执行:")
        print(inputs)
        print("-" * 50)
        
        # 这将调用 .env 中配置的真实 LLM
        result = crew_instance.kickoff(inputs=inputs)
        
        print("-" * 50)
        print("[测试] Crew 执行成功完成！")
        print("执行结果:")
        print(result)
        
    except Exception as e:
        print(f"\n[测试] 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
