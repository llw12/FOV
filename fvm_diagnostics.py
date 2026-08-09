"""Bounded, production-safe diagnostics for FVM RS frame failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fov import MetaPacket, SymbolPacket
from scripts.fvm0_rs_common import RSDecodeResult

RS_FAILURE_EVENT_LIMIT = 512


@dataclass(frozen=True)
class PacketDiagnostic:
    packet_type: str
    block_id: int | None = None
    symbol_id: int | None = None

    @classmethod
    def from_packet(cls, packet: MetaPacket | SymbolPacket | None) -> "PacketDiagnostic | None":
        if isinstance(packet, MetaPacket):
            return cls("META")
        if isinstance(packet, SymbolPacket):
            return cls("SYMBOL", packet.block_id, packet.symbol_id)
        return None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.packet_type}
        if self.packet_type == "SYMBOL":
            result.update({"block_id": self.block_id, "symbol_id": self.symbol_id})
        return result


@dataclass(frozen=True)
class SuccessfulTransport:
    transport_index: int
    packet: PacketDiagnostic | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport_index": self.transport_index,
            "packet": self.packet.to_dict() if self.packet is not None else None,
        }


@dataclass
class RSFailureEvent:
    observed_frame_index: int
    failed_codewords: int
    successful_codewords: int
    failed_codeword_indices: tuple[int, ...]
    corrections_per_codeword: tuple[int | None, ...]
    corrected_symbols_in_successful_codewords: int
    max_corrections_in_successful_codeword: int
    previous_successful: SuccessfulTransport | None
    next_successful: SuccessfulTransport | None = None
    inferred_transport_index: int | None = None
    inference_confident: bool = False

    def to_dict(self) -> dict[str, Any]:
        previous_packet = self.previous_successful.packet if self.previous_successful else None
        next_packet = self.next_successful.packet if self.next_successful else None
        same_block = (
            previous_packet is not None
            and next_packet is not None
            and previous_packet.packet_type == "SYMBOL"
            and next_packet.packet_type == "SYMBOL"
            and previous_packet.block_id == next_packet.block_id
        )
        return {
            "observed_frame_index": self.observed_frame_index,
            "failed_codewords": self.failed_codewords,
            "successful_codewords": self.successful_codewords,
            "failed_codeword_indices": list(self.failed_codeword_indices),
            "corrections_per_codeword": list(self.corrections_per_codeword),
            "corrected_symbols_in_successful_codewords": self.corrected_symbols_in_successful_codewords,
            "max_corrections_in_successful_codeword": self.max_corrections_in_successful_codeword,
            "previous_successful": self.previous_successful.to_dict() if self.previous_successful else None,
            "next_successful": self.next_successful.to_dict() if self.next_successful else None,
            "inferred_transport_index": self.inferred_transport_index,
            "inference_confident": self.inference_confident,
            "neighbor_block_consistent": same_block,
            "neighbor_block_id": previous_packet.block_id if same_block else None,
        }


class FailureTracker:
    """Track complete aggregate statistics and a bounded sample of detailed events."""

    def __init__(self, event_limit: int = RS_FAILURE_EVENT_LIMIT) -> None:
        if event_limit <= 0:
            raise ValueError("event_limit must be positive")
        self.event_limit = event_limit
        self.event_count_total = 0
        self.events: list[RSFailureEvent] = []
        self._last_successful: SuccessfulTransport | None = None
        self._pending_events: list[RSFailureEvent] = []
        self._pending_count = 0
        self._pending_inference_valid = True
        self._pending_first_observed_index: int | None = None
        self._pending_last_observed_index: int | None = None
        self._last_failure_index: int | None = None
        self._distance_count = 0
        self._distance_total = 0
        self._distance_min: int | None = None
        self._distance_max: int | None = None
        self._distance_sample: list[int] = []
        self._failed_codewords_total = 0
        self._failed_codewords_min: int | None = None
        self._failed_codewords_max = 0
        self._failed_codeword_histogram = [0] * 28
        self._burst_count = 0
        self._bursts: list[dict[str, int]] = []
        self._current_burst: dict[str, int] | None = None
        self._longest_burst = 0

    def observe_failure(self, observed_frame_index: int, rs: RSDecodeResult) -> None:
        self.event_count_total += 1
        self._record_distance(observed_frame_index)
        self._record_failed_codewords(rs)
        self._record_burst(observed_frame_index, rs.codeword_failures)
        event = RSFailureEvent(
            observed_frame_index=observed_frame_index,
            failed_codewords=rs.codeword_failures,
            successful_codewords=rs.codeword_successes,
            failed_codeword_indices=rs.failed_codeword_indices,
            corrections_per_codeword=rs.corrections_per_codeword,
            corrected_symbols_in_successful_codewords=rs.corrected_symbols,
            max_corrections_in_successful_codeword=rs.max_corrections_per_codeword,
            previous_successful=self._last_successful,
        )
        if len(self.events) < self.event_limit:
            self.events.append(event)
            self._pending_events.append(event)
        if self._pending_count == 0:
            self._pending_first_observed_index = observed_frame_index
        else:
            assert self._pending_last_observed_index is not None
            if observed_frame_index != self._pending_last_observed_index + 1:
                self._pending_inference_valid = False
        self._pending_last_observed_index = observed_frame_index
        self._pending_count += 1

    def observe_success(
        self,
        transport_index: int,
        packet: PacketDiagnostic | None,
    ) -> None:
        self._finish_burst()
        successful = SuccessfulTransport(transport_index, packet)
        if self._pending_count:
            self._close_pending(successful)
        self._last_successful = successful

    def observe_non_failure_without_transport(self) -> None:
        """End an RS burst and prevent inference across a non-transport frame."""
        self._finish_burst()
        if self._pending_count:
            self._pending_inference_valid = False

    def summary(self) -> dict[str, Any]:
        self._finish_burst()
        distance_mean = (
            self._distance_total / self._distance_count if self._distance_count else None
        )
        failed_mean = (
            self._failed_codewords_total / self.event_count_total
            if self.event_count_total else None
        )
        return {
            "event_count_total": self.event_count_total,
            "events_recorded": len(self.events),
            "events_truncated": self.event_count_total > len(self.events),
            "event_limit": self.event_limit,
            "events": [event.to_dict() for event in self.events],
            "observed_indices": [event.observed_frame_index for event in self.events],
            "inter_failure_distances": self._distance_sample,
            "inter_failure_distance_stats": {
                "count": self._distance_count,
                "min": self._distance_min,
                "max": self._distance_max,
                "mean": distance_mean,
            },
            "failed_codewords_total": self._failed_codewords_total,
            "failed_codewords_per_failed_frame": {
                "min": self._failed_codewords_min,
                "max": self._failed_codewords_max if self.event_count_total else None,
                "mean": failed_mean,
            },
            "failed_codeword_index_histogram": {
                str(index): count for index, count in enumerate(self._failed_codeword_histogram)
            },
            "consecutive_bursts": self._bursts,
            "burst_count": self._burst_count,
            "bursts_recorded": len(self._bursts),
            "bursts_truncated": self._burst_count > len(self._bursts),
            "longest_consecutive_burst": self._longest_burst,
        }

    def _record_distance(self, observed_frame_index: int) -> None:
        if self._last_failure_index is not None:
            distance = observed_frame_index - self._last_failure_index
            self._distance_count += 1
            self._distance_total += distance
            self._distance_min = distance if self._distance_min is None else min(self._distance_min, distance)
            self._distance_max = distance if self._distance_max is None else max(self._distance_max, distance)
            if len(self._distance_sample) < self.event_limit:
                self._distance_sample.append(distance)
        self._last_failure_index = observed_frame_index

    def _record_failed_codewords(self, rs: RSDecodeResult) -> None:
        self._failed_codewords_total += rs.codeword_failures
        self._failed_codewords_min = (
            rs.codeword_failures
            if self._failed_codewords_min is None
            else min(self._failed_codewords_min, rs.codeword_failures)
        )
        self._failed_codewords_max = max(self._failed_codewords_max, rs.codeword_failures)
        for index in rs.failed_codeword_indices:
            self._failed_codeword_histogram[index] += 1

    def _record_burst(self, observed_frame_index: int, failed_codewords: int) -> None:
        if self._current_burst is None:
            self._burst_count += 1
            self._current_burst = {
                "start_observed_frame": observed_frame_index,
                "end_observed_frame": observed_frame_index,
                "frame_count": 1,
                "total_failed_codewords": failed_codewords,
            }
            return
        if observed_frame_index == self._current_burst["end_observed_frame"] + 1:
            self._current_burst["end_observed_frame"] = observed_frame_index
            self._current_burst["frame_count"] += 1
            self._current_burst["total_failed_codewords"] += failed_codewords
            return
        self._finish_burst()
        self._record_burst(observed_frame_index, failed_codewords)

    def _finish_burst(self) -> None:
        if self._current_burst is None:
            return
        self._longest_burst = max(self._longest_burst, self._current_burst["frame_count"])
        if len(self._bursts) < self.event_limit:
            self._bursts.append(self._current_burst)
        self._current_burst = None

    def _close_pending(self, next_successful: SuccessfulTransport) -> None:
        for event in self._pending_events:
            event.next_successful = next_successful
        previous_index = (
            self._last_successful.transport_index if self._last_successful is not None else None
        )
        if previous_index is None:
            confident = (
                self._pending_first_observed_index == 0
                and next_successful.transport_index == self._pending_count
            )
            first_inferred = 0
        else:
            confident = (
                next_successful.transport_index > previous_index
                and next_successful.transport_index - previous_index - 1 == self._pending_count
            )
            first_inferred = previous_index + 1
        confident = confident and self._pending_inference_valid
        if confident:
            for offset, event in enumerate(self._pending_events):
                event.inferred_transport_index = first_inferred + offset
                event.inference_confident = True
        self._pending_events.clear()
        self._pending_count = 0
        self._pending_inference_valid = True
        self._pending_first_observed_index = None
        self._pending_last_observed_index = None
