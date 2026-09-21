"""三阶互调产物分析：枚举 2*fa - fb，聚合来源并评估最近命中对象。"""
from __future__ import annotations

from .schemas import GuardBand, HitObject, ProductOut, RiskGroup, SourcePair


def enumerate_third_order_products(
    carriers_hz: list[int], scan_start_hz: int, scan_end_hz: int
) -> dict[int, set[tuple[int, int]]]:
    """枚举不同载频的三阶产物 2*fa - fb（fa != fb）。

    仅保留落在扫描范围 [scan_start_hz, scan_end_hz] 内的产物；
    同频产物聚合全部来源组合 (fa, fb)。
    """
    products: dict[int, set[tuple[int, int]]] = {}
    for fa in carriers_hz:
        for fb in carriers_hz:
            if fa == fb:
                continue
            freq = 2 * fa - fb
            if scan_start_hz <= freq <= scan_end_hz:
                products.setdefault(freq, set()).add((fa, fb))
    return products


def _distance_to_guard_band(freq: int, band: GuardBand) -> int:
    """到保护频段的距离：落在带内为 0，否则为到最近边界的距离。"""
    if band.start_hz <= freq <= band.end_hz:
        return 0
    return min(abs(freq - band.start_hz), abs(freq - band.end_hz))


def nearest_hit(
    freq: int, carriers_hz: list[int], guard_bands: list[GuardBand]
) -> tuple[int, HitObject]:
    """返回 freq 到载频或保护频段的最短距离及对应命中对象。

    距离相同时按确定性规则决胜：载频优先于保护频段，其次按参考频率
    （载频值或频段起点）升序，保证结果稳定可复现。
    """
    best: tuple[int, int, int, HitObject] | None = None
    for carrier in carriers_hz:
        candidate = (
            abs(freq - carrier),
            0,
            carrier,
            HitObject(type="carrier", carrier_hz=carrier),
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    for band in guard_bands:
        candidate = (
            _distance_to_guard_band(freq, band),
            1,
            band.start_hz,
            HitObject(type="guard_band", guard_band=band),
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    assert best is not None  # carriers_hz 至少含一个载频
    return best[0], best[3]


def analyze(
    carriers_hz: list[int],
    scan_start_hz: int,
    scan_end_hz: int,
    guard_bands: list[GuardBand],
    guard_interval_hz: int,
) -> list[ProductOut]:
    """完整分析：枚举产物 -> 过滤范围外 -> 同频聚合 -> 计算最近命中对象。

    返回按产物频率升序排列的结果（同频已聚合，频率唯一，排序稳定确定）。
    """
    grouped = enumerate_third_order_products(carriers_hz, scan_start_hz, scan_end_hz)
    results: list[ProductOut] = []
    for freq in sorted(grouped):
        distance, hit = nearest_hit(freq, carriers_hz, guard_bands)
        sources = [SourcePair(fa_hz=fa, fb_hz=fb) for fa, fb in sorted(grouped[freq])]
        results.append(
            ProductOut(
                freq_hz=freq,
                distance_hz=distance,
                hit=hit,
                violates_guard_interval=distance < guard_interval_hz,
                sources=sources,
            )
        )
    return results


def build_risk_groups(products: list[ProductOut]) -> list[RiskGroup]:
    """由已持久化的分析结果构建风险组（纯读取，不重新计算分析）。

    仅违规产物（violates_guard_interval）作为风险节点参与构图；每个风险节点
    与其全部来源组合中的 fa_hz、fb_hz 载频连边，形成“产物-载频”二部图，
    连通分量即风险组（共享载频传递归组）。同频产物的多来源合并到同一节点，
    非违规产物不参与。组按最小载频再按最小产物频率排序，group_id 自 1 连续编号。
    """
    # 同频合并：持久化结果本已按频率聚合，此处防御性再合并一次
    merged: dict[int, dict] = {}
    for p in products:
        if not p.violates_guard_interval:
            continue
        node = merged.setdefault(p.freq_hz, {"distance_hz": p.distance_hz, "sources": []})
        node["distance_hz"] = min(node["distance_hz"], p.distance_hz)
        node["sources"].extend(p.sources)

    # 并查集：节点为 ("p", 产物频率) 或 ("c", 载频)
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x: tuple[str, int]) -> tuple[str, int]:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # 路径压缩
            parent[x], x = root, parent[x]
        return root

    def link(a: tuple[str, int], b: tuple[str, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for freq, node in merged.items():
        pnode = ("p", freq)
        parent.setdefault(pnode, pnode)
        for s in node["sources"]:
            for carrier in (s.fa_hz, s.fb_hz):
                cnode = ("c", carrier)
                parent.setdefault(cnode, cnode)
                link(pnode, cnode)

    # 汇总连通分量：每个分量至少含一个产物节点，载频由产物节点的边带来
    comps: dict[tuple[str, int], dict] = {}
    for freq, node in merged.items():
        comp = comps.setdefault(
            find(("p", freq)),
            {"carriers": set(), "freqs": [], "pairs": 0, "distance_hz": node["distance_hz"]},
        )
        comp["freqs"].append(freq)
        comp["pairs"] += len(node["sources"])
        comp["distance_hz"] = min(comp["distance_hz"], node["distance_hz"])
    for key in parent:
        if key[0] == "c":
            comps[find(key)]["carriers"].add(key[1])

    groups = [
        {
            "carriers_hz": sorted(c["carriers"]),
            "product_freqs_hz": sorted(c["freqs"]),
            "source_pair_count": c["pairs"],
            "distance_hz": c["distance_hz"],
        }
        for c in comps.values()
    ]
    # 产物存在必有来源，故每组载频非空；按最小载频再按最小产物频率排序
    groups.sort(key=lambda g: (g["carriers_hz"][0], g["product_freqs_hz"][0]))
    return [RiskGroup(group_id=i, **g) for i, g in enumerate(groups, start=1)]
