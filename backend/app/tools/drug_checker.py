"""
药物相互作用检查工具 (Drug Interaction Checker)

检查药物组合的安全性,防止危险的药物相互作用
"""

from typing import Dict, Any, List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from app.tools.base import BaseTool

logger = structlog.get_logger()


class DrugInteractionChecker(BaseTool):
    """
    药物相互作用检查工具
    
    用途:
    - 医生开具处方时检查药物安全性
    - 识别危险的药物组合
    - 提供用药建议
    
    数据源:
    1. 本地规则库(快速,覆盖常见组合)
    2. 外部API(慢,准确,覆盖全面) [未来实现]
    """
    
    def __init__(self):
        super().__init__()
        self.local_rules = self._load_local_rules()
    
    def _load_local_rules(self) -> Dict[Tuple[str, ...], Dict]:
        """
        加载本地药物相互作用规则库
        
        格式: {
            (药物A, 药物B): {
                'severity': 'high|medium|low',
                'warning': '警告信息',
                'mechanism': '作用机制'
            }
        }
        """
        return {
            # 高风险组合
            ('华法林', '阿司匹林'): {
                'severity': 'high',
                'warning': '增加出血风险,需密切监测INR值',
                'mechanism': '双重抗凝作用'
            },
            ('华法林', '阿斯匹林'): {  # 别名
                'severity': 'high',
                'warning': '增加出血风险,需密切监测INR值',
                'mechanism': '双重抗凝作用'
            },
            
            ('头孢曲松', '酒精'): {
                'severity': 'high',
                'warning': '可能引发双硫仑样反应(面部潮红、心悸、呼吸困难)',
                'mechanism': '抑制乙醛脱氢酶'
            },
            ('头孢', '酒精'): {  # 简称
                'severity': 'high',
                'warning': '可能引发双硫仑样反应,用药期间及停药后1周内应避免饮酒',
                'mechanism': '抑制乙醛脱氢酶'
            },
            
            ('地高辛', '氢氯噻嗪'): {
                'severity': 'medium',
                'warning': '利尿剂可能导致低钾,增加地高辛毒性风险',
                'mechanism': '电解质紊乱增加心脏毒性'
            },
            
            ('氨氯地平', '西柚汁'): {
                'severity': 'medium',
                'warning': '西柚汁抑制药物代谢,可能导致血压过低',
                'mechanism': 'CYP3A4抑制'
            },
            
            # 中等风险
            ('阿司匹林', '布洛芬'): {
                'severity': 'medium',
                'warning': '增加胃肠道出血风险,建议间隔用药',
                'mechanism': '双重抑制COX酶'
            },
            
            ('甲氨蝶呤', '阿司匹林'): {
                'severity': 'high',
                'warning': '增加甲氨蝶呤毒性,可能导致骨髓抑制',
                'mechanism': '竞争性抑制肾小管分泌'
            },
        }
    
    @property
    def description(self) -> str:
        return """
        检查药物相互作用,识别危险的药物组合。
        
        参数:
        - drugs (list[str]): 药品列表,例如 ['阿司匹林', '华法林']
        
        返回:
        - has_interaction: 是否存在相互作用
        - interactions: 相互作用详情列表
        - severity: 最高严重等级
        - source: 数据来源
        """
    
    async def _execute(self, drugs: List[str]) -> Dict[str, Any]:
        """
        执行药物相互作用检查
        
        流程:
        1. 标准化药品名称
        2. 检查本地规则库
        3. (可选)调用外部API
        4. 汇总结果
        """
        # 1. 标准化药品名称(去除空格、统一大小写)
        drugs = [d.strip() for d in drugs]
        
        # 2. 检查本地规则库
        interactions = []
        max_severity = 'none'
        
        # 两两组合检查
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                drug_pair = tuple(sorted([drugs[i], drugs[j]]))
                
                # 正向查找
                if drug_pair in self.local_rules:
                    interaction = self.local_rules[drug_pair]
                    interactions.append({
                        'drugs': list(drug_pair),
                        'severity': interaction['severity'],
                        'warning': interaction['warning'],
                        'mechanism': interaction['mechanism']
                    })
                    
                    # 更新最高严重等级
                    if interaction['severity'] == 'high':
                        max_severity = 'high'
                    elif interaction['severity'] == 'medium' and max_severity != 'high':
                        max_severity = 'medium'
                
                # 反向查找(处理别名)
                else:
                    reverse_pair = tuple([drugs[j], drugs[i]])
                    if reverse_pair in self.local_rules:
                        interaction = self.local_rules[reverse_pair]
                        interactions.append({
                            'drugs': list(reverse_pair),
                            'severity': interaction['severity'],
                            'warning': interaction['warning'],
                            'mechanism': interaction['mechanism']
                        })
                        
                        if interaction['severity'] == 'high':
                            max_severity = 'high'
                        elif interaction['severity'] == 'medium' and max_severity != 'high':
                            max_severity = 'medium'
        
        # 3. 格式化结果
        return {
            'success': True,
            'data': {
                'has_interaction': len(interactions) > 0,
                'interactions': interactions,
                'severity': max_severity,
                'checked_drugs': drugs,
                'count': len(interactions)
            },
            'confidence': 0.9,  # 本地规则库高置信度
            'source': 'local_db'
        }
    
    def _fallback(self, error: Exception, **kwargs) -> Dict[str, Any]:
        """
        降级方案: 保守策略
        
        任何检查失败时,建议咨询医生
        """
        drugs = kwargs.get('drugs', [])
        
        logger.warning(
            "drug_checker.fallback",
            drugs=drugs,
            error=str(error)
        )
        
        return {
            'success': False,
            'data': {
                'has_interaction': None,  # 未知
                'interactions': [{
                    'drugs': drugs,
                    'severity': 'unknown',
                    'warning': '药物相互作用检查暂时不可用,建议咨询药剂师或医生',
                    'mechanism': 'N/A'
                }],
                'severity': 'unknown',
                'checked_drugs': drugs,
                'count': 0
            },
            'confidence': 0.0,
            'source': 'fallback'
        }


# 便捷函数
async def check_drug_interaction(drugs: List[str]) -> Dict[str, Any]:
    """
    快捷调用药物检查
    
    示例:
        result = await check_drug_interaction(['阿司匹林', '华法林'])
        if result['data']['has_interaction']:
            print(f"警告: {result['data']['interactions'][0]['warning']}")
    """
    tool = DrugInteractionChecker()
    return await tool.run(drugs=drugs)
