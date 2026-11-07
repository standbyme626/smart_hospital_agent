from app.retrieval import rrf_fuse, tokenize, vector_literal


def test_tokenize_chinese():
    tokens = tokenize("高血压患者能吃阿司匹林吗")
    assert "高血压" in tokens
    assert "阿司匹林" in tokens


def test_rrf_prefers_shared_item():
    fused = rrf_fuse([["a", "b"], ["b", "c"]])
    assert fused[0][0] == "b"


def test_vector_literal():
    assert vector_literal([0.1, 0.2]) == "[0.100000,0.200000]"
