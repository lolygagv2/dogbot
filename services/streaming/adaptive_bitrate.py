#!/usr/bin/env python3
"""
Adaptive bitrate / resolution controller for WIM-Z WebRTC streaming.

aiortc has no browser-style `setParameters({encodings})` API and its VP8 encoder
hard-caps `target_bitrate` at 1.5 Mbps. This controller works within those real
limits:

  - Resolution is changed via WIMZVideoTrack.set_output_resolution() — the codec
    does not scale, so we resize frames ourselves.
  - Bitrate is changed via the VP8 encoder's `target_bitrate` property.
  - The network signal is packet loss + RTT (from pc.getStats(), populated by
    RTCP receiver reports) plus REMB observed indirectly: aiortc applies the
    receiver's REMB estimate to encoder.target_bitrate, so the value read back
    before we re-assert our tier cap reflects the receiver's bandwidth estimate.

Three tiers, all 15 fps. Fresh connections start at Low and adapt upward.
Adaptation is mode-agnostic — the network decides the tier, not drive/default.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable, List


@dataclass(frozen=True)
class Tier:
    """One quality tier: resolution + bitrate ceiling."""
    name: str
    width: int
    height: int
    bitrate: int  # bits per second

    @property
    def resolution(self) -> tuple:
        return (self.width, self.height)


# Ordered low -> high. High bitrate == aiortc's VP8 MAX_BITRATE (1.5 Mbps).
TIERS: List[Tier] = [
    Tier("low",     640, 480,  400_000),
    Tier("medium",  960, 540,  900_000),
    Tier("high",   1280, 720, 1_500_000),
]
_TIER_BY_NAME = {t.name: i for i, t in enumerate(TIERS)}


class AdaptiveBitrateController:
    """Per-connection adaptive bitrate + resolution controller."""

    LOOP_INTERVAL = 2.5          # seconds between stats checks
    STEP_UP_HOLD = 10.0          # seconds of sustained "good" before stepping up
    CHANGE_COOLDOWN = 5.0        # min seconds between any two tier changes

    # Network thresholds
    LOSS_BAD = 0.05              # >5% packet loss -> step down
    LOSS_GOOD = 0.02             # <2% loss is part of "good"
    RTT_BAD = 0.5                # >500ms RTT -> step down
    RTT_GOOD = 0.3               # <300ms RTT is part of "good"
    BITRATE_CONSTRAINED = 0.80   # observed/cap below this -> REMB says step down
    BITRATE_HEADROOM = 0.95      # observed/cap at/above this -> REMB has headroom

    def __init__(self, session_id: str, pc, video_track, logger=None,
                 on_change: Optional[Callable] = None):
        self.session_id = session_id
        self.pc = pc
        self.video_track = video_track
        self.logger = logger or logging.getLogger("AdaptiveBitrate")
        # Optional callback(status_dict) fired on every tier change — wired by
        # the app-quality contract layer to push the indicator to the app.
        self.on_change = on_change

        self._tier_idx = 0                        # start at Low
        self._manual_tier: Optional[str] = None   # None = auto/adaptive
        self._good_since: Optional[float] = None
        self._last_change = 0.0
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Latest measured network stats (for get_status / app indicator)
        self._loss = 0.0
        self._rtt = 0.0
        # True once at least one RTCP receiver report has arrived. Until then
        # loss/rtt are just their 0.0 initial values — a never-connected session
        # looks "perfect" and must not step up (or show bars) on that.
        self._media_confirmed = False

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        self.logger.info(
            f"[ABR] {self.session_id}: controller started (tier={self.current_tier.name})"
        )
        # Apply the starting tier immediately so the stream opens at Low.
        self._apply_tier(self.current_tier, reason="start")

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self.logger.info(f"[ABR] {self.session_id}: controller stopped")

    # --- public API ------------------------------------------------------

    @property
    def current_tier(self) -> Tier:
        return TIERS[self._tier_idx]

    def set_manual_override(self, mode: str) -> None:
        """Pin a tier or return to adaptive. mode: 'auto'|'low'|'medium'|'high'."""
        mode = (mode or "auto").lower()
        if mode == "auto":
            self._manual_tier = None
            self.logger.info(f"[ABR] {self.session_id}: quality mode -> auto")
        elif mode in _TIER_BY_NAME:
            self._manual_tier = mode
            self._tier_idx = _TIER_BY_NAME[mode]
            self.logger.info(f"[ABR] {self.session_id}: quality mode -> manual {mode}")
            self._apply_tier(self.current_tier, reason="manual")
        else:
            self.logger.warning(f"[ABR] {self.session_id}: unknown quality mode '{mode}'")

    def get_status(self) -> dict:
        """Current quality state for the app indicator."""
        t = self.current_tier
        return {
            "tier": t.name,
            "resolution": f"{t.width}x{t.height}",
            "bitrate_kbps": t.bitrate // 1000,
            "loss_pct": round(self._loss * 100, 1),
            "rtt_ms": round(self._rtt * 1000),
            "bars": self._bars(),
            "mode": "manual" if self._manual_tier else "auto",
        }

    # --- internals -------------------------------------------------------

    def _bars(self) -> int:
        """Connection-quality score 0-4 from loss + RTT, for the app indicator."""
        if not self._media_confirmed:
            return 0
        score = 4
        if self._loss > 0.02:
            score -= 1
        if self._loss > 0.08:
            score -= 1
        if self._rtt > 0.3:
            score -= 1
        if self._rtt > 0.6:
            score -= 1
        return max(0, min(4, score))

    def _video_sender(self):
        for s in self.pc.getSenders():
            if s.track is not None and s.track.kind == "video":
                return s
        return None

    def _encoder(self):
        """The VP8 encoder — created lazily by aiortc on the first encoded frame."""
        sender = self._video_sender()
        if sender is None:
            return None
        return getattr(sender, "_RTCRtpSender__encoder", None)

    def _set_encoder_bitrate(self, bitrate: int) -> None:
        enc = self._encoder()
        if enc is not None and hasattr(enc, "target_bitrate"):
            enc.target_bitrate = bitrate

    def _apply_tier(self, tier: Tier, reason: str) -> None:
        # Resolution — always safe to set on the track.
        self.video_track.set_output_resolution(tier.resolution)
        # Bitrate — encoder may not exist yet (first frame not encoded); guarded.
        self._set_encoder_bitrate(tier.bitrate)
        self._last_change = time.monotonic()
        self.logger.info(
            f"[ABR] {self.session_id}: tier -> {tier.name} "
            f"({tier.width}x{tier.height} @ {tier.bitrate // 1000}kbps) [{reason}]"
        )
        if self.on_change:
            try:
                self.on_change(self.get_status())
            except Exception as e:
                self.logger.debug(f"[ABR] on_change callback error: {e}")

    async def _read_stats(self):
        """Return (loss_fraction 0-1, rtt_seconds, observed_bitrate or None)."""
        loss, rtt = self._loss, self._rtt
        try:
            report = await self.pc.getStats()
            for stat in report.values():
                # RTCRemoteInboundRtpStreamStats = the receiver's report on our
                # outbound stream (loss + RTT), delivered via RTCP.
                if type(stat).__name__ == "RTCRemoteInboundRtpStreamStats":
                    self._media_confirmed = True
                    fl = getattr(stat, "fractionLost", None)
                    if fl is not None:
                        # aiortc reports RTCP fraction-lost as a raw 0-255 byte;
                        # normalise to a 0-1 fraction if needed.
                        loss = fl / 256.0 if fl > 1.0 else fl
                    rtt_val = getattr(stat, "roundTripTime", None)
                    if rtt_val:
                        rtt = rtt_val
        except Exception as e:
            self.logger.debug(f"[ABR] getStats error: {e}")

        # Observed bitrate = what aiortc's REMB handling left target_bitrate at.
        observed = None
        enc = self._encoder()
        if enc is not None and hasattr(enc, "target_bitrate"):
            observed = enc.target_bitrate
        return loss, rtt, observed

    async def _run(self) -> None:
        ticks = 0
        try:
            while self._running:
                await asyncio.sleep(self.LOOP_INTERVAL)
                if not self._running:
                    break
                loss, rtt, observed = await self._read_stats()
                self._loss, self._rtt = loss, rtt
                # Periodic INFO trace of the raw step-up/down inputs — without
                # this the journal can't answer "why is it stuck at low?"
                ticks += 1
                if ticks % 4 == 0:
                    obs_k = f"{observed // 1000}k" if observed else "none"
                    self.logger.info(
                        f"[ABR] {self.session_id}: stats loss={loss:.1%} "
                        f"rtt={rtt * 1000:.0f}ms observed={obs_k} "
                        f"cap={self.current_tier.bitrate // 1000}k "
                        f"media_confirmed={self._media_confirmed} "
                        f"tier={self.current_tier.name}"
                    )
                # Adaptive decision — skipped under manual override.
                if self._manual_tier is None:
                    self._evaluate(loss, rtt, observed)
                # Status heartbeat every tick so the app indicator stays
                # current (bars/loss/rtt) even without a tier change.
                if self.on_change:
                    try:
                        self.on_change(self.get_status())
                    except Exception as e:
                        self.logger.debug(f"[ABR] status callback error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"[ABR] {self.session_id}: control loop error: {e}")

    def _evaluate(self, loss: float, rtt: float, observed) -> None:
        now = time.monotonic()
        cap = self.current_tier.bitrate

        # REMB signal: aiortc pulled target_bitrate below our cap -> constrained.
        remb_constrained = observed is not None and observed < cap * self.BITRATE_CONSTRAINED
        remb_headroom = observed is None or observed >= cap * self.BITRATE_HEADROOM

        bad = (loss > self.LOSS_BAD) or (rtt > self.RTT_BAD) or remb_constrained
        good = (self._media_confirmed and loss < self.LOSS_GOOD
                and rtt < self.RTT_GOOD and remb_headroom)

        # If aiortc's REMB pushed target_bitrate ABOVE our tier cap (good network
        # but our tier wants a lower ceiling), re-assert the cap.
        if observed is not None and observed > cap:
            self._set_encoder_bitrate(cap)

        # Step down immediately on bad network.
        if bad and self._tier_idx > 0:
            if now - self._last_change >= self.CHANGE_COOLDOWN:
                self._tier_idx -= 1
                self._good_since = None
                self._apply_tier(
                    self.current_tier,
                    reason=f"step-down (loss={loss:.1%} rtt={rtt * 1000:.0f}ms)")
            return

        # Step up only after sustained good conditions.
        if good and self._tier_idx < len(TIERS) - 1:
            if self._good_since is None:
                self._good_since = now
            elif (now - self._good_since >= self.STEP_UP_HOLD
                  and now - self._last_change >= self.CHANGE_COOLDOWN):
                self._tier_idx += 1
                self._good_since = None
                self._apply_tier(self.current_tier, reason="step-up (sustained good)")
        else:
            self._good_since = None
