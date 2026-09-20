"""IPv4 数据报分片重组核心逻辑（纯标准库实现，无 Web 依赖）。

输入 IPv4 首部长度（20～60 字节范围内 4 的倍数，即 20/24/…/60，不含在
重组结果内）与同一数据报的 1～500 个分片，每个分片给出：

- ``offset``：片偏移字段值，以 **8 字节** 为单位；
- ``more_fragments``：MF 标志，是否还有后续分片；
- ``payload``：本分片承载的载荷字节。

重组规则：

1. **非末片对齐**：除唯一一个 ``more_fragments=false`` 的末片外，其余
   分片的载荷长度必须是 8 的倍数（IPv4 片偏移以 8 字节计）。
2. **末片定长**：必须**恰好一个**末片；末片按
   ``offset*8 + len(payload)`` 确定总载荷长度。
3. **数据报上限**：总载荷长度加首部长度不能越过 IPv4 数据报 65535 字节
   上限。
4. **逐片边界（裁剪前）**：每片 ``offset`` 必须是片偏移字段可表达的
   0～8191（8 字节单位），且
   ``offset*8 + len(payload) + header_bytes`` 不得越过 65535。该校验对
   **每个**分片执行——即使分片完全落在末片锚定总长度之外（随后仍会被
   裁剪掉）也不例外，使畸形分片无法借“落在锚点外”被静默忽略。
5. **按字节区间重组**：每个分片占据半开字节区间
   ``[offset*8, offset*8 + len(payload))``。重叠覆盖处字节必须一致，
   重复且字节一致的覆盖可接受；任一重叠字节冲突即报 ``conflict`` 与
   **首个冲突偏移**（全部分片中最小的不一致字节偏移），优先于缺口结论。
6. 无冲突但相对 ``[0, 总载荷长度)`` 存在缺口时返回 ``incomplete`` 与
   合并后的缺失半开区间；覆盖完整时返回 ``complete``、按偏移排序的
   **贡献分片索引**（至少贡献一个总载荷字节的输入分片序号）与重组后的
   Base64 载荷。

实现上先完成首部、末片数量、非末片对齐、末片锚定总长度与**逐片偏移/
边界**全部校验，再把每个分片裁剪成锚点内的半开字节区间，然后对按起点
排序的区间做
**扫描线**：在覆盖集不变的每个段内只做切片级比较（C 层 memcmp / memcpy，
二分定位段内首个不一致偏移），既不为每个分片逐字节遍历，也不构造与总载荷
等长的 owner 列表；段按偏移升序产出，故任意输入顺序下都能直接得到全局
最小冲突偏移，且冲突判定先于缺口计算。
"""
from __future__ import annotations

import base64
import binascii
from bisect import insort
from typing import Sequence

from .analyzer import merge_intervals, subtract_covered

#: IPv4 首部长度（IHL，单位字节）取值范围
HEADER_MIN = 20
HEADER_MAX = 60

#: IHL 以 32 位字（4 字节）为单位，合法首部长度必须是 4 的倍数
HEADER_UNIT = 4

#: 合法 IPv4 首部长度枚举（20～60 内的全部 4 字节倍数）；非法值统一在
#: 422 detail.allowed_values 中原样返回（元组 JSON 序列化后为数组）
HEADER_ALLOWED = tuple(range(HEADER_MIN, HEADER_MAX + 1, HEADER_UNIT))

#: 单次请求允许的分片数区间
FRAGMENTS_MIN = 1
FRAGMENTS_MAX = 500

#: IPv4 数据报（首部 + 载荷）总长度上限
IPV4_DATAGRAM_MAX = 65535

#: 片偏移字段以 8 字节为单位
OFFSET_UNIT = 8

#: IPv4 片偏移字段为 13 位，最大值（以 8 字节为单位）。0～8191 的片偏移
#: 是每个重组输入分片都必须满足的字段级约束（fragmentation 模块共用）
FRAGMENT_OFFSET_MAX = 8191

#: 重组结论
COMPLETE = "complete"
INCOMPLETE = "incomplete"
CONFLICT = "conflict"

#: 输入错误类型（Web 层映射为 422 detail.error）
ERR_INVALID_HEADER_LENGTH = "invalid_header_length"
ERR_INVALID_FRAGMENT_OFFSET = "invalid_fragment_offset"
ERR_NON_ALIGNED_FRAGMENT = "non_aligned_fragment"
ERR_LAST_FRAGMENT_COUNT = "invalid_last_fragment_count"
ERR_DATAGRAM_TOO_LARGE = "datagram_too_large"


class ReassemblyInputError(ValueError):
    """分片重组输入不合法（Web 层映射为 422）。

    - ``error``：错误类型标识（``invalid_header_length`` /
      ``invalid_fragment_offset`` / ``non_aligned_fragment`` /
      ``invalid_last_fragment_count`` / ``datagram_too_large``）；
    - ``indexes``：相关分片的输入序号（末片数量类问题给出全部末片序号，
      偏移 / 对齐 / 超长类问题给出首个违规分片；与首部相关时为 ``[]``）；
    - ``limit``：越界错误附带的上限值（总长度类问题为 65535，其余为
      ``None``）；
    - ``header_bytes`` / ``allowed_values``：``invalid_header_length``
      专用，分别给出实际首部长度与全部合法取值（20～60 内 4 的倍数），
      其余错误为 ``None``。
    """

    def __init__(
        self,
        error: str,
        message: str,
        indexes: list[int] | None = None,
        limit: int | None = None,
        header_bytes: object = None,
        allowed_values: tuple[int, ...] | None = None,
    ) -> None:
        self.error = error
        self.indexes = indexes or []
        self.limit = limit
        self.header_bytes = header_bytes
        self.allowed_values = list(allowed_values) if allowed_values is not None else None
        super().__init__(message)


def is_valid_header_length(header_length: object) -> bool:
    """合法 IPv4 首部长度判定：20～60 范围内 4 的倍数的非布尔整数。

    分片重组与分片规划两个核心函数以及 HTTP 入口必须共用本判定，保证
    同一套取值规则（21、22、23 等不是 4 字节倍数的值一律非法）。
    """
    return (
        not isinstance(header_length, bool)
        and isinstance(header_length, int)
        and HEADER_MIN <= header_length <= HEADER_MAX
        and header_length % HEADER_UNIT == 0
    )


def validate_header_length(header_length: object) -> int:
    """校验首部长度：20～60 范围内 4 的倍数（拒绝 bool 与浮点等）。

    这是两个核心函数与 HTTP 入口共用的唯一规则入口；非法即抛出
    :class:`ReassemblyInputError`（``invalid_header_length``，携带实际
    ``header_bytes`` 与完整 ``allowed_values``）。
    """
    if not is_valid_header_length(header_length):
        if isinstance(header_length, bool) or not isinstance(header_length, int):
            message = (
                f"header_bytes 必须是 {HEADER_MIN}～{HEADER_MAX} 范围内 "
                f"{HEADER_UNIT} 的倍数的整数: {header_length!r}"
            )
        elif not HEADER_MIN <= header_length <= HEADER_MAX:
            message = (
                f"header_bytes 必须在 {HEADER_MIN}～{HEADER_MAX} 之间且为 "
                f"{HEADER_UNIT} 的倍数，实际 {header_length}"
            )
        else:
            message = (
                f"header_bytes 必须是 {HEADER_UNIT} 字节的倍数（IHL 以 "
                f"{HEADER_UNIT} 字节为单位），实际 {header_length}"
            )
        raise ReassemblyInputError(
            ERR_INVALID_HEADER_LENGTH,
            message,
            header_bytes=header_length,
            allowed_values=HEADER_ALLOWED,
        )
    return header_length


def decode_payload(value: object) -> bytes:
    """解码标准 Base64 载荷；空串视为空字节串。

    使用 ``validate=True`` 拒绝任何填充错误或非字母表字符（含内嵌空白），
    非法 Base64 抛出 :class:`ValueError`。
    """
    if not isinstance(value, str):
        raise ValueError("payload 必须是 Base64 字符串")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"非法 Base64 payload: {exc}") from exc


def _validate_common(
    header_length: object,
    fragments: Sequence[tuple[object, object, object]],
) -> tuple[int, list[dict]]:
    """校验首部长度与分片列表规模，并逐片解析为字节区间表示。

    返回 ``(header_length, parsed_fragments)``；后者每项为
    ``{"index", "offset", "offset_bytes", "more_fragments",
    "payload"}``，其中 ``offset`` 为原始 8 字节单位片偏移。

    首部长度校验（20～60 内 4 的倍数）先于末片、对齐、逐片边界与 65535
    上限等全部后续判定，使多重非法输入得到稳定错误。片偏移的类型 / 负数
    等结构性输入错误在这里以 :class:`ValueError` 抛出（Web 层通常已在入口
    拦截）；片偏移越过 13 位字段上限 8191 等**语义**越界由
    :func:`reassemble_ipv4` 在末片数量、非末片对齐与末片锚定总长度校验
    之后、锚点裁剪之前统一以 :class:`ReassemblyInputError` 报告。
    """
    header_length = validate_header_length(header_length)

    if not isinstance(fragments, Sequence) or isinstance(fragments, (str, bytes)):
        raise ValueError("fragments 必须是分片元组列表")
    if not FRAGMENTS_MIN <= len(fragments) <= FRAGMENTS_MAX:
        raise ValueError(
            f"分片数必须在 {FRAGMENTS_MIN}～{FRAGMENTS_MAX} 之间，"
            f"实际 {len(fragments)}"
        )

    parsed: list[dict] = []
    for index, fragment in enumerate(fragments):
        if len(fragment) != 3:
            raise ValueError("每个分片必须是 (offset, more_fragments, payload)")
        raw_offset, more_fragments, payload = fragment
        if isinstance(raw_offset, bool) or not isinstance(raw_offset, int):
            raise ValueError(f"分片 {index} 的 offset 必须是非负整数: {raw_offset!r}")
        if raw_offset < 0:
            raise ValueError(f"分片 {index} 的 offset 不能为负: {raw_offset}")
        if not isinstance(more_fragments, bool):
            raise ValueError(
                f"分片 {index} 的 more_fragments 必须是布尔值: {more_fragments!r}"
            )
        if isinstance(payload, str):
            payload = decode_payload(payload)
        if not isinstance(payload, bytes):
            raise ValueError(f"分片 {index} 的 payload 必须是 Base64 字符串或字节串")
        parsed.append(
            {
                "index": index,
                "offset": raw_offset,
                "offset_bytes": raw_offset * OFFSET_UNIT,
                "more_fragments": more_fragments,
                "payload": payload,
            }
        )
    return header_length, parsed


def reassemble_ipv4(
    header_length: object,
    fragments: Sequence[tuple[object, object, object]],
) -> dict:
    """重组同一 IPv4 数据报的分片，返回可直接 JSON 序列化的结果字典。

    ``fragments`` 每项为 ``(offset, more_fragments, payload)``，其中
    ``offset`` 为 8 字节单位的非负整数、``more_fragments`` 为布尔值、
    ``payload`` 为字节串（独立使用时也接受标准 Base64 字符串）。

    首部长度必须为 20～60 范围内 4 的倍数（20/24/…/60）；该校验先于
    末片数量、非末片对齐、逐片边界与 65535 上限判定。语义校验按以下
    固定次序进行：末片数量必须恰好一个；非末片载荷必须 8 字节对齐；
    末片锚定的总载荷长度加首部不得越过 65535；随后在锚点裁剪前对
    **每个分片**（含完全落在锚点之外、随后会被整段裁剪掉的分片）校验
    片偏移在 0～8191 内，且
    ``offset*8 + len(payload) + header_length`` 不越过 65535。

    返回结构（按结论三选一，无多余键）：

    - ``{"status": "conflict", "first_conflict_offset": int}``：
      重叠覆盖处存在不一致字节，给出全部分片中最小的冲突字节偏移；
    - ``{"status": "incomplete", "missing_ranges": [[start, end], ...]}``：
      无冲突但 ``[0, 总载荷长度)`` 未被完全覆盖，区间为合并后的**半开**
      ``[start, end)``、按起点升序；
    - ``{"status": "complete", "contributing_fragments": [int, ...],
      "payload": "<Base64>"}``：覆盖完整，贡献分片按
      ``(字节偏移, 输入序号)`` 升序排列，载荷为重组后字节的标准 Base64。

    语义类输入错误抛出 :class:`ReassemblyInputError`（首部长度非法、
    片偏移越过 8191、非末片未按 8 字节对齐、末片数量不为一、末片锚定
    总长度或任一分片自身边界越过 65535）；
    非法 Base64、offset 类型/负数等抛出 :class:`ValueError`（Web 层
    通常已在入口拦截）。
    """
    header_length, parsed = _validate_common(header_length, fragments)

    # 末片数量校验必须先于总长度判定（没有末片即无从确定总载荷长度）
    last_indexes = [f["index"] for f in parsed if not f["more_fragments"]]
    if len(last_indexes) != 1:
        raise ReassemblyInputError(
            ERR_LAST_FRAGMENT_COUNT,
            f"必须恰好有一个 more_fragments=false 的末片，实际 {len(last_indexes)} 个",
            indexes=last_indexes,
        )
    last_index = last_indexes[0]

    # 非末片载荷必须按 8 字节对齐；末片（总长度由其锚定）不要求对齐
    for fragment in parsed:
        if fragment["more_fragments"] and len(fragment["payload"]) % OFFSET_UNIT != 0:
            raise ReassemblyInputError(
                ERR_NON_ALIGNED_FRAGMENT,
                f"非末片 {fragment['index']} 的载荷长度 {len(fragment['payload'])} "
                f"不是 {OFFSET_UNIT} 的倍数",
                indexes=[fragment["index"]],
            )

    last_fragment = parsed[last_index]
    total_length = last_fragment["offset_bytes"] + len(last_fragment["payload"])
    if header_length + total_length > IPV4_DATAGRAM_MAX:
        raise ReassemblyInputError(
            ERR_DATAGRAM_TOO_LARGE,
            f"重组数据报长度 {header_length + total_length} 越过 IPv4 "
            f"{IPV4_DATAGRAM_MAX} 字节上限",
            indexes=[last_index],
            limit=IPV4_DATAGRAM_MAX,
        )

    # 逐片边界校验（锚点裁剪前）：片偏移必须落在 13 位字段范围 0～8191，
    # 且本片自身的绝对结束位置 offset*8 + 载荷长度 + 首部长度不得越过
    # 65535。该校验对**每一片**生效——即使分片完全落在末片锚定总长度
    # 之外（下一步会被整段裁剪）也不例外，使畸形分片不能借“落在锚点外”
    # 被静默忽略。按输入序号报告首个违规分片。
    for fragment in parsed:
        index = fragment["index"]
        offset = fragment["offset"]
        if not 0 <= offset <= FRAGMENT_OFFSET_MAX:
            raise ReassemblyInputError(
                ERR_INVALID_FRAGMENT_OFFSET,
                f"分片 {index} 的 offset 必须在 0～{FRAGMENT_OFFSET_MAX} "
                f"之间（IPv4 片偏移字段为 13 位），实际 {offset}",
                indexes=[index],
            )
        fragment_end = header_length + fragment["offset_bytes"] + len(
            fragment["payload"]
        )
        if fragment_end > IPV4_DATAGRAM_MAX:
            raise ReassemblyInputError(
                ERR_DATAGRAM_TOO_LARGE,
                f"分片 {index} 的绝对结束位置 {fragment_end}（offset*8 + "
                f"载荷长度 {len(fragment['payload'])} + 首部长度 "
                f"{header_length}）越过 IPv4 {IPV4_DATAGRAM_MAX} 字节上限",
                indexes=[index],
                limit=IPV4_DATAGRAM_MAX,
            )

    # 区间化：将每个分片裁剪为半开字节区间 [start, end) 与对应载荷
    # （越出末片锚定总长度的部分不参与数据报重组）。零长度 / 完全越界的
    # 分片保留在 parsed 中用于错误校验，但不产生 span。后续只对载荷做
    # **切片级**比较（C 层 memcmp；CPython 的 memoryview 相等比较反而走
    # 通用迭代路径，故这里保留 bytes 切片）。
    spans: list[tuple[int, int, int, bytes]] = []
    for fragment in parsed:
        start = fragment["offset_bytes"]
        end = min(start + len(fragment["payload"]), total_length)
        if start < end:
            spans.append((start, end, fragment["index"], fragment["payload"]))

    # 重叠冲突检测：按区间起点做扫描线。每个非平凡常量覆盖段 [lo, hi) 内，
    # 覆盖该段的分片集合不变；任取一个代表片，其余分片只需与代表片做
    # **切片级**比较（C 层 memcmp，不逐字节遍历）。段按偏移升序产生，首个
    # 含冲突的段即给出全局最小冲突偏移，与分片输入顺序无关。
    events: list[tuple[int, int, int]] = []
    for event_order, (start, end, _index, _payload) in enumerate(spans):
        events.append((start, 1, event_order))  # 起点：加入活动集
        events.append((end, 0, event_order))    # 终点：移出活动集
    events.sort()

    first_conflict: int | None = None
    active: list[int] = []  # 活动 span 在 spans 中的序号，按输入序号有序
    prev = 0
    event_idx = 0
    while event_idx < len(events):
        position = events[event_idx][0]
        lo, hi = prev, position
        if lo < hi and len(active) > 1:
            rep = min(active, key=lambda span_idx: spans[span_idx][2])
            rep_start, _rep_end, _rep_index, rep_payload = spans[rep]
            rep_seg = rep_payload[lo - rep_start:hi - rep_start]
            segment_conflict: int | None = None
            for span_idx in active:
                span_start, _span_end, _index, payload = spans[span_idx]
                segment = payload[lo - span_start:hi - span_start]
                if segment == rep_seg:
                    continue
                # 二分定位本片与代表片在该段内首个不一致字节：等长前缀在
                # 长度 0..d 上一致（d 为段内首个不一致相对偏移），故最小不
                # 一致前缀长度 lo_len = d+1；仅 O(log n) 次切片比较。
                lo_len, hi_len = 0, len(rep_seg)
                while lo_len < hi_len:
                    mid = (lo_len + hi_len) // 2
                    if segment[:mid] == rep_seg[:mid]:
                        lo_len = mid + 1
                    else:
                        hi_len = mid
                candidate = lo + lo_len - 1
                if segment_conflict is None or candidate < segment_conflict:
                    segment_conflict = candidate
            if segment_conflict is not None:
                first_conflict = segment_conflict
                break
        # 应用该位置上的全部事件（同位置先移出后加入，不影响下一段活动集）
        while event_idx < len(events) and events[event_idx][0] == position:
            _pos, kind, ev_order = events[event_idx]
            if kind == 1:
                insort(active, ev_order)
            else:
                active.remove(ev_order)
            event_idx += 1
        prev = position

    if first_conflict is not None:
        return {"status": CONFLICT, "first_conflict_offset": first_conflict}

    # 已覆盖区间（闭区间）归并后求其在 [0, total_length) 内的补集
    filled = [(start, end - 1) for start, end, _index, _payload in spans]
    missing_closed = (
        subtract_covered((0, total_length - 1), merge_intervals(filled))
        if total_length
        else []
    )
    missing_ranges = [[start, end + 1] for start, end in missing_closed]
    if missing_ranges:
        return {"status": INCOMPLETE, "missing_ranges": missing_ranges}

    # 覆盖完整且无冲突：重叠处字节必然一致，按 (偏移, 输入序号) 序以切片
    # 写入重组缓冲（C 层 memcpy，不逐字节拷贝）；仅写入落在锚定长度内的片。
    contributing = sorted(
        {index for _start, _end, index, _payload in spans},
        key=lambda index: (parsed[index]["offset_bytes"], index),
    )
    buffer = bytearray(total_length)
    for start, end, _index, payload in sorted(
        spans, key=lambda span: (span[0], span[2])
    ):
        buffer[start:end] = payload[: end - start]
    return {
        "status": COMPLETE,
        "contributing_fragments": contributing,
        "payload": base64.b64encode(bytes(buffer)).decode("ascii"),
    }
