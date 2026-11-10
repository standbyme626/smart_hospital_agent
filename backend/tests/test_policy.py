from app.policy import DRUG_CHECKER, detect_crisis, rule_triage


def test_crisis_detection():
    matched, terms = detect_crisis("胸痛伴出汗，呼吸困难，意识不清")
    assert matched is True
    assert "呼吸困难" in terms


def test_crisis_negative():
    matched, terms = detect_crisis("最近有点轻微头痛，想问问挂什么科")
    assert matched is False
    assert terms == []


def test_triage_pediatrics():
    result = rule_triage("孩子发烧 38.5 度，一直哭")
    assert result["department"] == "儿科"


def test_triage_crisis_override():
    result = rule_triage("父亲突然昏迷，没有呼吸")
    assert result["intent"] == "CRISIS"
    assert result["urgency"] == "critical"
    assert result["department"] == "急诊科"


def test_crisis_triage_carries_rule_fields():
    result = rule_triage("我爸刚才晕倒了")
    assert result["rule_triggered"] is True
    assert result["crisis_subtype"] == "stroke_acs"
    assert "晕倒" in result["crisis_terms"]


def test_drug_interaction_major():
    result = DRUG_CHECKER.check_text("患者在吃华法林，现在牙痛想吃阿司匹林")
    assert "华法林" in result["drugs"]
    assert "阿司匹林" in result["drugs"]
    assert result["highest_severity"] == "major"
    assert result["interactions"]
    assert result["coverage"] == "full"
    assert result["exposures"] == []
    assert result["unrecognized"] == []


def test_pregnancy_contraindication():
    result = DRUG_CHECKER.check_text("孕妇可以用异维A酸吗")
    assert result["pregnancy_contraindications"]
    assert result["coverage"] == "full"


def test_pregnancy_contraindication_structured_flag():
    result = DRUG_CHECKER.check_text("可以用华法林吗", pregnant=True)
    assert result["pregnancy_contraindications"]
    result = DRUG_CHECKER.check_text("孕妇可以用华法林吗", pregnant=False)
    assert result["pregnancy_contraindications"] == []
