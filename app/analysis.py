"""三阶互调产物分析：枚举 2*fa - fb，聚合来源并评估最近命中对象。"""
from __future__ import annotations

from .schemas import GuardBand, HitObject, ProductOut, SourcePair


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
