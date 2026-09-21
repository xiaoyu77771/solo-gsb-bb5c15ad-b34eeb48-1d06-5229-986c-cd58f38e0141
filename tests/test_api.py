"""API 端到端测试：使用临时 SQLite 库，通过 TestClient 走完整请求链路。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("IMD_DB_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:  # 触发 lifespan -> init_db
        yield c


def make_payload(**overrides):
    payload = {
        "name": "site-a",
        "scan_start_hz": 1,
        "scan_end_hz": 10_000,
        "guard_interval_hz": 150,
        "carriers_hz": [100, 300, 400],
        "guard_bands": [],
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------- 站点创建与校验

def test_create_site_ok(client):
    resp = client.post("/sites", json=make_payload())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["id"] == 1
    assert body["carriers_hz"] == [100, 300, 400]
    assert body["guard_interval_hz"] == 150
    # 查询已保存方案
    assert client.get(f"/sites/{body['id']}").json()["name"] == "site-a"
    assert len(client.get("/sites").json()) == 1


def test_duplicate_carriers_rejected_as_whole_batch(client):
    resp = client.post("/sites", json=make_payload(carriers_hz=[100, 200, 100]))
    assert resp.status_code == 422
    assert client.get("/sites").json() == []  # 整批拒绝，未写入任何数据


def test_invalid_guard_band_rejected_as_whole_batch(client):
    bad = make_payload(guard_bands=[{"start_hz": 500, "end_hz": 500}])
    assert client.post("/sites", json=bad).status_code == 422
    bad2 = make_payload(guard_bands=[{"start_hz": 900, "end_hz": 100}])
    assert client.post("/sites", json=bad2).status_code == 422
    assert client.get("/sites").json() == []


def test_invalid_scan_range_rejected(client):
    assert client.post("/sites", json=make_payload(scan_end_hz=1)).status_code == 422
    assert client.get("/sites").json() == []


def test_non_positive_values_rejected(client):
    assert client.post("/sites", json=make_payload(carriers_hz=[0, 100])).status_code == 422
    assert client.post("/sites", json=make_payload(guard_interval_hz=-1)).status_code == 422
    assert client.get("/sites").json() == []


def test_duplicate_site_name_conflict(client):
    assert client.post("/sites", json=make_payload()).status_code == 201
    resp = client.post("/sites", json=make_payload())
    assert resp.status_code == 409
    sites = client.get("/sites").json()
    assert len(sites) == 1  # 第二个方案未残留


def test_get_missing_site_404(client):
    assert client.get("/sites/999").status_code == 404


# ---------------------------------------------------------------- 互调分析

def test_analysis_aggregates_same_frequency_sources(client):
    # 载频 [100, 300, 400]：2*300-100=500 与 2*400-300=500 同频，须聚合
    client.post("/sites", json=make_payload())
    resp = client.post("/sites/1/analysis")
    assert resp.status_code == 201, resp.text
    body = resp.json()

    freqs = [p["freq_hz"] for p in body["products"]]
    assert freqs == sorted(freqs)  # 按频率稳定排序
    assert freqs == [200, 500, 700]  # 范围外产物（<=0）已被过滤

    p500 = next(p for p in body["products"] if p["freq_hz"] == 500)
    assert {tuple(s.values()) for s in p500["sources"]} == {(300, 100), (400, 300)}

    # 命中对象与最短距离
    p200 = next(p for p in body["products"] if p["freq_hz"] == 200)
    assert p200["distance_hz"] == 100
    assert p200["hit"] == {"type": "carrier", "carrier_hz": 100, "guard_band": None}
    assert p200["violates_guard_interval"] is True  # 100 < 150

    p700 = next(p for p in body["products"] if p["freq_hz"] == 700)
    assert p700["hit"]["carrier_hz"] == 400
    assert p700["distance_hz"] == 300
    assert p700["violates_guard_interval"] is False  # 300 >= 150


def test_analysis_hits_guard_band_with_zero_distance(client):
    payload = make_payload(
        carriers_hz=[1000, 1100],
        guard_bands=[{"start_hz": 880, "end_hz": 920}],
        guard_interval_hz=50,
    )
    client.post("/sites", json=payload)
    body = client.post("/sites/1/analysis").json()

    products = {p["freq_hz"]: p for p in body["products"]}
    assert set(products) == {900, 1200}

    # 900 落在保护频段 [880, 920] 内：距离 0，命中频段
    assert products[900]["distance_hz"] == 0
    assert products[900]["hit"]["type"] == "guard_band"
    assert products[900]["hit"]["guard_band"] == {"start_hz": 880, "end_hz": 920}
    assert products[900]["violates_guard_interval"] is True

    # 1200 距载频 1100 为 100，距频段最近边界 280 -> 命中载频
    assert products[1200]["hit"] == {"type": "carrier", "carrier_hz": 1100, "guard_band": None}
    assert products[1200]["distance_hz"] == 100


def test_analysis_filters_out_of_scan_range(client):
    payload = make_payload(carriers_hz=[200, 500], scan_start_hz=500, scan_end_hz=900)
    client.post("/sites", json=payload)
    body = client.post("/sites/1/analysis").json()
    # 2*200-500=-100 与 2*500-200=800：仅 800 在 [500, 900] 内
    assert [p["freq_hz"] for p in body["products"]] == [800]
    assert body["product_count"] == 1


def test_analysis_persisted_and_queryable(client):
    client.post("/sites", json=make_payload())
    created = client.post("/sites/1/analysis").json()

    latest = client.get("/sites/1/analysis").json()
    assert latest == created  # 查询已保存的分析结果

    by_id = client.get(f"/analyses/{created['analysis_id']}").json()
    assert by_id == created

    metas = client.get("/sites/1/analyses").json()
    assert [m["analysis_id"] for m in metas] == [created["analysis_id"]]
    assert metas[0]["product_count"] == created["product_count"]


def test_analysis_404s(client):
    assert client.post("/sites/999/analysis").status_code == 404
    assert client.get("/sites/999/analysis").status_code == 404
    assert client.get("/analyses/999").status_code == 404
    client.post("/sites", json=make_payload())
    assert client.get("/sites/1/analysis").status_code == 404  # 尚未分析


# ---------------------------------------------------------------- 风险组

def test_risk_groups_merge_multi_source_same_frequency(client):
    # 载频 [100, 300, 400]：违规产物 200{(300,400)} 与 500{(300,100),(400,300)}
    # 同频产物 500 的多来源合并到同一节点，并经共享载频与 200 归为一组
    client.post("/sites", json=make_payload())
    analysis_id = client.post("/sites/1/analysis").json()["analysis_id"]

    resp = client.get(f"/analyses/{analysis_id}/risk-groups")
    assert resp.status_code == 200, resp.text
    assert resp.json() == [
        {
            "group_id": 1,
            "carriers_hz": [100, 300, 400],
            "product_freqs_hz": [200, 500],
            "source_pair_count": 3,  # 500 的两个来源组合都计入
            "distance_hz": 100,  # min(100, 100)
        }
    ]


def test_risk_groups_chain_shared_carriers_and_exclude_non_violating(client):
    # 载频 [100, 200, 400, 1000, 1100]，guard_interval=250：
    # 违规产物 300{200,100}(d=100)、600{400,200}(d=200) 经共享载频 200 链式归组；
    # 900{1000,1100}、1200{1100,1000} 为另一组。
    # 非违规产物 700{400,100}、1800{(1000,200),(1100,400)} 等不参与，
    # 否则两组会被桥接合并。
    payload = make_payload(carriers_hz=[100, 200, 400, 1000, 1100], guard_interval_hz=250)
    client.post("/sites", json=payload)
    analysis_id = client.post("/sites/1/analysis").json()["analysis_id"]

    resp = client.get(f"/analyses/{analysis_id}/risk-groups")
    assert resp.status_code == 200, resp.text
    assert resp.json() == [
        {  # 按最小载频排序：100 < 1000，group_id 自 1 连续编号
            "group_id": 1,
            "carriers_hz": [100, 200, 400],
            "product_freqs_hz": [300, 600],
            "source_pair_count": 2,
            "distance_hz": 100,  # min(100, 200)
        },
        {
            "group_id": 2,
            "carriers_hz": [1000, 1100],
            "product_freqs_hz": [900, 1200],
            "source_pair_count": 2,
            "distance_hz": 100,
        },
    ]


def test_risk_groups_empty_when_no_violation(client):
    # guard_interval_hz=0：任何产物都不违规，返回空列表
    client.post("/sites", json=make_payload(guard_interval_hz=0))
    analysis_id = client.post("/sites/1/analysis").json()["analysis_id"]
    resp = client.get(f"/analyses/{analysis_id}/risk-groups")
    assert resp.status_code == 200
    assert resp.json() == []


def test_risk_groups_use_persisted_result(client):
    # 风险组基于已持久化的分析结果：同一 analysis_id 多次查询结果稳定一致
    payload = make_payload(
        carriers_hz=[1000, 1100],
        guard_bands=[{"start_hz": 880, "end_hz": 920}],
        guard_interval_hz=50,
    )
    client.post("/sites", json=payload)
    analysis_id = client.post("/sites/1/analysis").json()["analysis_id"]

    expected = [
        {  # 仅 900（落在保护频段内，距离 0）违规；1200 不违规不参与
            "group_id": 1,
            "carriers_hz": [1000, 1100],
            "product_freqs_hz": [900],
            "source_pair_count": 1,
            "distance_hz": 0,
        }
    ]
    assert client.get(f"/analyses/{analysis_id}/risk-groups").json() == expected
    assert client.get(f"/analyses/{analysis_id}/risk-groups").json() == expected


def test_risk_groups_404(client):
    assert client.get("/analyses/999/risk-groups").status_code == 404


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
