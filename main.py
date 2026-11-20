"""Real-time Driver / User Drowsiness & Fatigue Detection System.

Features:
    - 6-Point Eye Aspect Ratio (EAR) calculation using Euclidean distance.
    - MediaPipe Face Mesh (468/478 landmarks) with refined eyelid geometry.
    - Non-blocking Audio Alert Engine with Pygame mixer + automatic NumPy buzzer synthesis fallback.
    - Interactive Eyelid Calibration (press 'c' to auto-calibrate baseline open-eye EAR).
    - Optional Yawn Detection via Mouth Aspect Ratio (MAR) (press 'm' to toggle).
    - Production HUD Overlay with adaptive semi-transparent dashboard, eye contours, and flashing alert banners.
    - Command-line argument support for custom webcams, recorded dashcam video files, and thresholds.

Author: Senior Computer Vision & Python Engineer
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Suppress Pygame welcome banner
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DrowsinessDetector")


# ============================================================================
# SYSTEM CONFIGURATION & HYPERPARAMETERS
# ============================================================================
@dataclass
class Config:
    """System hyperparameters, video options, and facial landmark mappings."""

    # Video Capture
    CAMERA_INDEX: int = 0
    VIDEO_PATH: Optional[str] = None
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    WINDOW_TITLE: str = "Real-time Driver Drowsiness Detection System"

    # Drowsiness Decision Boundaries
    EAR_THRESHOLD: float = 0.22  # Eye Aspect Ratio threshold (closed eye < 0.22)
    CLOSED_DURATION_THRESHOLD: float = 2.0  # Seconds of continuous closure before alarm triggers

    # Yawn Detection (MAR - Mouth Aspect Ratio)
    ENABLE_YAWN_DETECTION: bool = True
    MAR_THRESHOLD: float = 0.65  # Mouth Aspect Ratio threshold (yawning > 0.65)
    YAWN_DURATION_THRESHOLD: float = 2.5  # Seconds of open mouth to count as fatigue yawn

    # Audio Engine
    ALARM_AUDIO_PATH: str = "alarm.wav"
    FALLBACK_SAMPLE_RATE: int = 44100
    FALLBACK_TONE_FREQ_1: int = 880  # A5 note (Hz)
    FALLBACK_TONE_FREQ_2: int = 1100  # C#6 note (Hz)
    FALLBACK_TONE_DURATION: float = 0.6  # Seconds per buzzer cycle

    # MediaPipe 6-Point EAR Landmark Indices
    # Left Eye: Horizontal (33, 133), Vertical P1-P2 (160, 144), Vertical P3-P4 (158, 153)
    # Mapping: p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
    LEFT_EYE_LANDMARKS: Tuple[int, int, int, int, int, int] = (
        33,   # p1 (Outer/Inner lateral corner)
        160,  # p2 (Upper eyelid 1)
        158,  # p3 (Upper eyelid 2)
        133,  # p4 (Opposite corner)
        153,  # p5 (Lower eyelid 2)
        144,  # p6 (Lower eyelid 1)
    )

    # Right Eye: Horizontal (362, 263), Vertical P1-P2 (385, 380), Vertical P3-P4 (387, 373)
    # Mapping: p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
    RIGHT_EYE_LANDMARKS: Tuple[int, int, int, int, int, int] = (
        362,  # p1
        385,  # p2
        387,  # p3
        263,  # p4
        373,  # p5
        380,  # p6
    )

    # Mouth Landmark Indices for Yawn (MAR) Calculation:
    # Horizontal corners (78, 308), Vertical pairs (13, 14), (82, 87), (312, 317)
    MOUTH_LANDMARKS: Tuple[int, int, int, int, int, int, int, int] = (
        78,   # Left corner
        308,  # Right corner
        13,   # Upper inner lip center
        14,   # Lower inner lip center
        82,   # Upper inner lip left
        87,   # Lower inner lip left
        312,  # Upper inner lip right
        317,  # Lower inner lip right
    )

    # Polygon boundary orders for eyelid and lip contours
    LEFT_EYE_CONTOUR: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE_CONTOUR: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)
    MOUTH_CONTOUR: Tuple[int, ...] = (78, 82, 13, 312, 308, 317, 14, 87)

    # UI Theme Palette (BGR format)
    COLOR_BG_CARD: Tuple[int, int, int] = (24, 24, 28)
    COLOR_TEXT_PRIMARY: Tuple[int, int, int] = (240, 240, 245)
    COLOR_TEXT_MUTED: Tuple[int, int, int] = (160, 160, 175)
    COLOR_NORMAL: Tuple[int, int, int] = (46, 204, 113)      # Emerald Green
    COLOR_WARNING: Tuple[int, int, int] = (0, 165, 255)      # Amber Orange
    COLOR_DROWSY: Tuple[int, int, int] = (30, 30, 235)       # Bright Crimson Red
    COLOR_EYE_CONTOUR: Tuple[int, int, int] = (255, 215, 0)  # Cyan-Gold Highlight
    COLOR_MOUTH_CONTOUR: Tuple[int, int, int] = (255, 140, 0)


# ============================================================================
# MODULE 1: MATHEMATICAL CORE (EAR & MAR CALCULATION)
# ============================================================================
class BiometricCalculator:
    """Computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) via Euclidean distance."""

    @staticmethod
    def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate standard 2D Euclidean distance between two pixel coordinates."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @classmethod
    def calculate_single_ear(
        cls,
        pts: Dict[int, Tuple[int, int]],
        p1_idx: int,
        p2_idx: int,
        p3_idx: int,
        p4_idx: int,
        p5_idx: int,
        p6_idx: int,
    ) -> float:
        """Compute Eye Aspect Ratio (EAR) for a 6-point landmark model.

        Formula:
            EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
        """
        p1 = pts[p1_idx]
        p2 = pts[p2_idx]
        p3 = pts[p3_idx]
        p4 = pts[p4_idx]
        p5 = pts[p5_idx]
        p6 = pts[p6_idx]

        v1 = cls.euclidean_distance(p2, p6)
        v2 = cls.euclidean_distance(p3, p5)
        h = cls.euclidean_distance(p1, p4)

        if h < 1e-6:
            return 0.0

        return (v1 + v2) / (2.0 * h)

    @classmethod
    def calculate_mar(
        cls,
        pts: Dict[int, Tuple[int, int]],
        mouth_indices: Tuple[int, int, int, int, int, int, int, int],
    ) -> float:
        """Compute Mouth Aspect Ratio (MAR) to detect prolonged yawning.

        Formula:
            MAR = (||p13 - p14|| + ||p82 - p87|| + ||p312 - p317||) / (2.0 * ||p78 - p308||)
        """
        p78, p308, p13, p14, p82, p87, p312, p317 = (pts[idx] for idx in mouth_indices)

        v_center = cls.euclidean_distance(p13, p14)
        v_left = cls.euclidean_distance(p82, p87)
        v_right = cls.euclidean_distance(p312, p317)
        h = cls.euclidean_distance(p78, p308)

        if h < 1e-6:
            return 0.0

        return (v_center + v_left + v_right) / (2.0 * h)


# Backward compatibility alias
EarCalculator = BiometricCalculator


# ============================================================================
# MODULE 2: NON-BLOCKING AUDIO ENGINE
# ============================================================================
class AudioAlertEngine:
    """Thread-safe, non-blocking audio engine using Pygame Mixer with NumPy buzzer synthesis."""

    def __init__(self, audio_path: str = Config.ALARM_AUDIO_PATH):
        self.audio_path = audio_path
        self._is_playing: bool = False
        self._lock = threading.Lock()
        self._sound: Optional[pygame.mixer.Sound] = None
        self._channel: Optional[pygame.mixer.Channel] = None
        self._initialized: bool = False

        self._init_mixer()

    def _init_mixer(self) -> None:
        """Initialize Pygame Mixer and load or synthesize alarm sound."""
        try:
            pygame.mixer.init(
                frequency=Config.FALLBACK_SAMPLE_RATE,
                size=-16,
                channels=2,
                buffer=512,
            )
            self._channel = pygame.mixer.Channel(0)
            self._initialized = True
            logger.info("Pygame mixer initialized successfully.")
        except Exception as exc:
            logger.warning(
                f"Failed to initialize Pygame mixer (Audio disabled): {exc}"
            )
            self._initialized = False
            return

        if os.path.exists(self.audio_path):
            try:
                self._sound = pygame.mixer.Sound(self.audio_path)
                logger.info(f"Loaded external alarm audio from: {self.audio_path}")
                return
            except Exception as exc:
                logger.warning(
                    f"Could not load audio file '{self.audio_path}': {exc}. Generating fallback tone."
                )

        self._sound = self._synthesize_buzzer_sound()

    def _synthesize_buzzer_sound(self) -> Optional[pygame.mixer.Sound]:
        """Synthesize a dual-tone pulsed buzzer sound via NumPy array buffer."""
        try:
            sr = Config.FALLBACK_SAMPLE_RATE
            duration = Config.FALLBACK_TONE_DURATION
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)

            # Pulsed dual-tone: Beep1 (880Hz) -> Pause -> Beep2 (1100Hz) -> Pause
            wave = np.zeros_like(t)
            beep1 = (t >= 0.0) & (t < 0.15)
            wave[beep1] = 0.55 * np.sin(2 * np.pi * Config.FALLBACK_TONE_FREQ_1 * t[beep1])

            beep2 = (t >= 0.22) & (t < 0.40)
            wave[beep2] = 0.65 * np.sin(2 * np.pi * Config.FALLBACK_TONE_FREQ_2 * t[beep2])

            pcm16 = np.int16(wave * 32767)
            stereo = np.ascontiguousarray(np.column_stack((pcm16, pcm16)))

            sound = pygame.sndarray.make_sound(stereo)
            logger.info("Synthesized high-urgency fallback buzzer via NumPy.")
            return sound
        except Exception as exc:
            logger.error(f"Fallback sound synthesis failed: {exc}")
            return None

    def start_alarm(self) -> None:
        """Trigger continuous alarm playback in background non-blocking channel."""
        with self._lock:
            if not self._initialized or self._sound is None:
                return
            if not self._is_playing:
                try:
                    self._channel.play(self._sound, loops=-1)
                    self._is_playing = True
                    logger.info("Alarm playback started.")
                except Exception as exc:
                    logger.error(f"Error starting alarm playback: {exc}")

    def stop_alarm(self) -> None:
        """Halt alarm playback immediately."""
        with self._lock:
            if not self._initialized:
                return
            if self._is_playing:
                try:
                    if self._channel is not None:
                        self._channel.stop()
                    self._is_playing = False
                    logger.info("Alarm playback halted.")
                except Exception as exc:
                    logger.error(f"Error stopping alarm: {exc}")

    @property
    def is_playing(self) -> bool:
        """Return current playback status."""
        with self._lock:
            return self._is_playing

    def release(self) -> None:
        """Safely shutdown audio mixer resources."""
        self.stop_alarm()
        try:
            pygame.mixer.quit()
            logger.info("Audio engine shut down cleanly.")
        except Exception:
            pass


# ============================================================================
# MODULE 3: STATE MACHINE & DETECTOR LOGIC
# ============================================================================
class DetectionState:
    NORMAL: str = "NORMAL"
    DROWSY: str = "DROWSY"
    YAWNING: str = "YAWNING"
    NO_FACE: str = "NO_FACE"


@dataclass
class DrowsinessStatus:
    """Snapshot of current system state and metrics for a frame."""

    state: str
    ear_avg: float
    ear_left: float
    ear_right: float
    mar: float
    closed_duration: float
    closed_frames: int
    yawn_duration: float
    is_alarm_active: bool
    eye_pts_left: Dict[int, Tuple[int, int]]
    eye_pts_right: Dict[int, Tuple[int, int]]
    mouth_pts: Dict[int, Tuple[int, int]]


class DrowsinessDetector:
    """State machine tracking Eye Aspect Ratio, Yawn Duration, and alert triggers."""

    def __init__(
        self,
        ear_threshold: float = Config.EAR_THRESHOLD,
        duration_threshold: float = Config.CLOSED_DURATION_THRESHOLD,
        mar_threshold: float = Config.MAR_THRESHOLD,
        yawn_duration_threshold: float = Config.YAWN_DURATION_THRESHOLD,
        enable_yawn: bool = Config.ENABLE_YAWN_DETECTION,
        audio_engine: Optional[AudioAlertEngine] = None,
    ):
        self.ear_threshold: float = ear_threshold
        self.duration_threshold: float = duration_threshold
        self.mar_threshold: float = mar_threshold
        self.yawn_duration_threshold: float = yawn_duration_threshold
        self.enable_yawn: bool = enable_yawn
        self.audio_engine: Optional[AudioAlertEngine] = audio_engine

        self.current_state: str = DetectionState.NORMAL
        self.closed_start_time: Optional[float] = None
        self.closed_frames_count: int = 0
        self.latest_closed_duration: float = 0.0

        self.yawn_start_time: Optional[float] = None
        self.latest_yawn_duration: float = 0.0

    def update(
        self,
        ear_left: Optional[float],
        ear_right: Optional[float],
        mar: Optional[float],
        pts_left: Dict[int, Tuple[int, int]],
        pts_right: Dict[int, Tuple[int, int]],
        pts_mouth: Dict[int, Tuple[int, int]],
    ) -> DrowsinessStatus:
        """Update state machine with newest frame metrics."""
        now = time.time()

        if ear_left is None or ear_right is None:
            self._reset_eyes_closed()
            self._reset_yawn()
            if self.audio_engine:
                self.audio_engine.stop_alarm()
            self.current_state = DetectionState.NO_FACE
            return DrowsinessStatus(
                state=self.current_state,
                ear_avg=0.0,
                ear_left=0.0,
                ear_right=0.0,
                mar=0.0,
                closed_duration=0.0,
                closed_frames=0,
                yawn_duration=0.0,
                is_alarm_active=False,
                eye_pts_left={},
                eye_pts_right={},
                mouth_pts={},
            )

        ear_avg = (ear_left + ear_right) / 2.0
        current_mar = mar if mar is not None else 0.0

        # --------------------------------------------------------------------
        # 1. Evaluate Eyes Closure (EAR)
        # --------------------------------------------------------------------
        is_eyes_closed = ear_avg < self.ear_threshold
        if is_eyes_closed:
            if self.closed_start_time is None:
                self.closed_start_time = now
            self.closed_frames_count += 1
            self.latest_closed_duration = now - self.closed_start_time
        else:
            self._reset_eyes_closed()

        # --------------------------------------------------------------------
        # 2. Evaluate Yawning (MAR)
        # --------------------------------------------------------------------
        is_yawning = self.enable_yawn and (current_mar >= self.mar_threshold)
        if is_yawning:
            if self.yawn_start_time is None:
                self.yawn_start_time = now
            self.latest_yawn_duration = now - self.yawn_start_time
        else:
            self._reset_yawn()

        # --------------------------------------------------------------------
        # 3. Decision Logic & State Transitions
        # --------------------------------------------------------------------
        if self.latest_closed_duration >= self.duration_threshold:
            self.current_state = DetectionState.DROWSY
            if self.audio_engine and not self.audio_engine.is_playing:
                self.audio_engine.start_alarm()
        elif self.latest_yawn_duration >= self.yawn_duration_threshold:
            self.current_state = DetectionState.YAWNING
            if self.audio_engine and not self.audio_engine.is_playing:
                self.audio_engine.start_alarm()
        else:
            self.current_state = DetectionState.NORMAL
            if self.audio_engine and self.audio_engine.is_playing:
                self.audio_engine.stop_alarm()

        return DrowsinessStatus(
            state=self.current_state,
            ear_avg=ear_avg,
            ear_left=ear_left,
            ear_right=ear_right,
            mar=current_mar,
            closed_duration=self.latest_closed_duration,
            closed_frames=self.closed_frames_count,
            yawn_duration=self.latest_yawn_duration,
            is_alarm_active=bool(
                self.audio_engine.is_playing if self.audio_engine else False
            ),
            eye_pts_left=pts_left,
            eye_pts_right=pts_right,
            mouth_pts=pts_mouth,
        )

    def _reset_eyes_closed(self) -> None:
        """Reset continuous eye closure counters."""
        self.closed_start_time = None
        self.closed_frames_count = 0
        self.latest_closed_duration = 0.0

    def _reset_yawn(self) -> None:
        """Reset yawn timers."""
        self.yawn_start_time = None
        self.latest_yawn_duration = 0.0


# ============================================================================
# MODULE 4: HUD & VISUAL FEEDBACK OVERLAY (OPENCV CANVAS)
# ============================================================================
class HUDOverlay:
    """Renders real-time HUD dashboard, dynamic eyelid/mouth contours, and visual alerts."""

    @staticmethod
    def draw_contours(
        frame: np.ndarray,
        pts: Dict[int, Tuple[int, int]],
        contour_indices: Tuple[int, ...],
        color: Tuple[int, int, int],
        point_color: Tuple[int, int, int] = (0, 255, 255),
    ) -> None:
        """Render polygon contour and keypoints with antialiased lines."""
        if not pts:
            return

        polygon_pts = np.array(
            [pts[idx] for idx in contour_indices if idx in pts], dtype=np.int32
        )
        if len(polygon_pts) >= 3:
            cv2.polylines(
                frame,
                [polygon_pts],
                isClosed=True,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        for idx in contour_indices:
            if idx in pts:
                cv2.circle(
                    frame,
                    pts[idx],
                    radius=2,
                    color=point_color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

    @classmethod
    def render_hud(
        cls,
        frame: np.ndarray,
        status: DrowsinessStatus,
        fps: float,
        detector: DrowsinessDetector,
        calibration_progress: Optional[float] = None,
    ) -> None:
        """Draw semi-transparent HUD card, telemetry stats, and blinking warning banners."""
        h, w, _ = frame.shape

        # --------------------------------------------------------------------
        # 1. Full-screen flashing alert border when drowsy or excessive yawning
        # --------------------------------------------------------------------
        if status.state in (DetectionState.DROWSY, DetectionState.YAWNING):
            is_flash_visible = int(time.time() * 3.5) % 2 == 0
            if is_flash_visible:
                alert_color = (
                    Config.COLOR_DROWSY
                    if status.state == DetectionState.DROWSY
                    else Config.COLOR_WARNING
                )
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color=alert_color, thickness=10)

        # --------------------------------------------------------------------
        # 2. Semi-transparent Dashboard Panel (Top-Left)
        # --------------------------------------------------------------------
        panel_w, panel_h = 460, 230
        panel_x, panel_y = 20, 20

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            color=Config.COLOR_BG_CARD,
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            color=(80, 80, 95),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "DRIVER MONITORING SYSTEM // HUD",
            (panel_x + 16, panel_y + 28),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.55,
            color=Config.COLOR_TEXT_MUTED,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # --------------------------------------------------------------------
        # 3. Dynamic Status Badge
        # --------------------------------------------------------------------
        badge_y = panel_y + 68
        if calibration_progress is not None:
            # Active Calibration Banner
            pct = int(calibration_progress * 100)
            cv2.putText(
                frame,
                f"CALIBRATING BASELINE... ({pct}%)",
                (panel_x + 16, badge_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.68,
                color=(0, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        elif status.state == DetectionState.DROWSY:
            is_blink = int(time.time() * 4) % 2 == 0
            cv2.putText(
                frame,
                "ALERT: DROWSINESS DETECTED!",
                (panel_x + 16, badge_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.70,
                color=(255, 255, 255) if is_blink else Config.COLOR_DROWSY,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        elif status.state == DetectionState.YAWNING:
            cv2.putText(
                frame,
                "WARNING: FATIGUE YAWN DETECTED!",
                (panel_x + 16, badge_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.68,
                color=Config.COLOR_WARNING,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        elif status.state == DetectionState.NO_FACE:
            cv2.putText(
                frame,
                "STATUS: NO FACE DETECTED",
                (panel_x + 16, badge_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.70,
                color=Config.COLOR_WARNING,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "STATUS: FOCUSED",
                (panel_x + 16, badge_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.72,
                color=Config.COLOR_NORMAL,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        # --------------------------------------------------------------------
        # 4. Telemetry Metrics
        # --------------------------------------------------------------------
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        thick = 1

        line1 = f"EAR Avg: {status.ear_avg:.2f}  [L: {status.ear_left:.2f} | R: {status.ear_right:.2f}]  (Thresh: {detector.ear_threshold:.2f})"
        line2 = f"MAR (Mouth): {status.mar:.2f}  (Thresh: {detector.mar_threshold:.2f} | Yawn: {'ON' if detector.enable_yawn else 'OFF'})"
        line3 = f"Eyes Closed: {status.closed_duration:.2f}s  ({status.closed_frames} frames)  [Limit: {detector.duration_threshold:.1f}s]"
        line4 = f"Yawn Time: {status.yawn_duration:.2f}s  [Limit: {detector.yawn_duration_threshold:.1f}s]"
        line5 = f"Engine: {fps:4.1f} FPS  |  Alarm: {'ACTIVE (PLAYING)' if status.is_alarm_active else 'STANDBY'}"

        cv2.putText(frame, line1, (panel_x + 16, panel_y + 104), font, scale, Config.COLOR_TEXT_PRIMARY, thick, cv2.LINE_AA)
        cv2.putText(frame, line2, (panel_x + 16, panel_y + 128), font, scale, Config.COLOR_TEXT_PRIMARY, thick, cv2.LINE_AA)

        closed_color = Config.COLOR_DROWSY if status.closed_duration > 0 else Config.COLOR_TEXT_MUTED
        cv2.putText(frame, line3, (panel_x + 16, panel_y + 152), font, scale, closed_color, thick, cv2.LINE_AA)

        yawn_color = Config.COLOR_WARNING if status.yawn_duration > 0 else Config.COLOR_TEXT_MUTED
        cv2.putText(frame, line4, (panel_x + 16, panel_y + 176), font, scale, yawn_color, thick, cv2.LINE_AA)

        cv2.putText(frame, line5, (panel_x + 16, panel_y + 202), font, scale, Config.COLOR_TEXT_PRIMARY, thick, cv2.LINE_AA)

        # --------------------------------------------------------------------
        # 5. Instructions Bar
        # --------------------------------------------------------------------
        footer_text = "[Q]/[ESC] Exit  |  [+] / [-] Sensitivity  |  [C] Auto-Calibrate  |  [M] Toggle Yawn"
        cv2.putText(frame, footer_text, (24, h - 20), font, 0.48, (190, 190, 190), 1, cv2.LINE_AA)


# ============================================================================
# MODULE 5: APPLICATION ORCHESTRATOR
# ============================================================================
class DrowsinessDetectorApp:
    """Coordinates camera capture, MediaPipe FaceMesh processing, and GUI loop."""

    def __init__(self, config: Config = Config()):
        self.cfg = config
        self.audio_engine = AudioAlertEngine(audio_path=self.cfg.ALARM_AUDIO_PATH)
        self.detector = DrowsinessDetector(
            ear_threshold=self.cfg.EAR_THRESHOLD,
            duration_threshold=self.cfg.CLOSED_DURATION_THRESHOLD,
            mar_threshold=self.cfg.MAR_THRESHOLD,
            yawn_duration_threshold=self.cfg.YAWN_DURATION_THRESHOLD,
            enable_yawn=self.cfg.ENABLE_YAWN_DETECTION,
            audio_engine=self.audio_engine,
        )
        self.camera: Optional[cv2.VideoCapture] = None

        # Auto-calibration state
        self._is_calibrating: bool = False
        self._calibration_start_time: float = 0.0
        self._calibration_samples: List[float] = []
        self._calibration_duration: float = 3.0  # seconds

    def _init_camera(self) -> cv2.VideoCapture:
        """Initialize camera or recorded video file stream."""
        if self.cfg.VIDEO_PATH:
            logger.info(f"Loading video file: {self.cfg.VIDEO_PATH}")
            cap = cv2.VideoCapture(self.cfg.VIDEO_PATH)
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video file at: {self.cfg.VIDEO_PATH}")
            return cap

        logger.info(f"Connecting to camera index {self.cfg.CAMERA_INDEX}...")
        cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX)

        if not cap.isOpened() and sys.platform.startswith("win"):
            logger.warning("Retrying camera with cv2.CAP_DSHOW backend...")
            cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX, cv2.CAP_DSHOW)

        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open video device at index {self.cfg.CAMERA_INDEX}. "
                "Ensure a webcam is connected and not locked by another process."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.FRAME_HEIGHT)
        return cap

    def trigger_calibration(self) -> None:
        """Start auto-calibration procedure to sample user's resting open-eye EAR."""
        logger.info("Initiating 3-second eye baseline calibration. Look straight with eyes naturally open.")
        self._is_calibrating = True
        self._calibration_start_time = time.time()
        self._calibration_samples = []

    def _process_calibration(self, ear_avg: Optional[float]) -> Optional[float]:
        """Collect samples and update threshold upon calibration completion."""
        if not self._is_calibrating:
            return None

        elapsed = time.time() - self._calibration_start_time
        progress = min(1.0, elapsed / self._calibration_duration)

        if ear_avg is not None and ear_avg > 0.15:
            self._calibration_samples.append(ear_avg)

        if progress >= 1.0:
            self._is_calibrating = False
            if self._calibration_samples:
                baseline_ear = float(np.mean(self._calibration_samples))
                # Set threshold to 72% of natural open-eye baseline
                new_threshold = round(baseline_ear * 0.72, 2)
                self.detector.ear_threshold = max(0.15, min(0.35, new_threshold))
                logger.info(
                    f"Calibration complete! Baseline EAR: {baseline_ear:.2f} -> New EAR Threshold: {self.detector.ear_threshold:.2f}"
                )
            else:
                logger.warning("Calibration incomplete: No face/eye samples detected.")
            return None

        return progress

    def run(self) -> None:
        """Main application lifecycle execution loop."""
        try:
            self.camera = self._init_camera()
        except RuntimeError as err:
            logger.critical(str(err))
            self.audio_engine.release()
            return

        logger.info("Initializing MediaPipe Face Mesh model...")
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        prev_time = time.time()
        fps = 30.0

        logger.info("System operational. Controls: [Q/ESC] Quit, [C] Calibrate, [M] Yawn Toggle, [+/-] Threshold")

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    # If reading video file, loop back to start
                    if self.cfg.VIDEO_PATH:
                        self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.warning("Camera frame read error or disconnected.")
                    break

                # Mirror frame horizontally for intuitive driver mirror experience (webcam only)
                if not self.cfg.VIDEO_PATH:
                    frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape

                # MediaPipe inference requires RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True

                ear_left: Optional[float] = None
                ear_right: Optional[float] = None
                mar: Optional[float] = None
                pts_left: Dict[int, Tuple[int, int]] = {}
                pts_right: Dict[int, Tuple[int, int]] = {}
                pts_mouth: Dict[int, Tuple[int, int]] = {}

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0].landmark

                    # Left Eye Landmarks
                    for idx in self.cfg.LEFT_EYE_LANDMARKS:
                        lm = face_landmarks[idx]
                        pts_left[idx] = (int(lm.x * w), int(lm.y * h))

                    # Right Eye Landmarks
                    for idx in self.cfg.RIGHT_EYE_LANDMARKS:
                        lm = face_landmarks[idx]
                        pts_right[idx] = (int(lm.x * w), int(lm.y * h))

                    # Mouth Landmarks for Yawn
                    for idx in self.cfg.MOUTH_LANDMARKS:
                        lm = face_landmarks[idx]
                        pts_mouth[idx] = (int(lm.x * w), int(lm.y * h))

                    ear_left = BiometricCalculator.calculate_single_ear(
                        pts_left, *self.cfg.LEFT_EYE_LANDMARKS
                    )
                    ear_right = BiometricCalculator.calculate_single_ear(
                        pts_right, *self.cfg.RIGHT_EYE_LANDMARKS
                    )
                    mar = BiometricCalculator.calculate_mar(
                        pts_mouth, self.cfg.MOUTH_LANDMARKS
                    )

                # Process ongoing calibration if active
                calib_ear = (
                    (ear_left + ear_right) / 2.0
                    if (ear_left is not None and ear_right is not None)
                    else None
                )
                calibration_progress = self._process_calibration(calib_ear)

                # Update Drowsiness State Machine
                status = self.detector.update(
                    ear_left=ear_left,
                    ear_right=ear_right,
                    mar=mar,
                    pts_left=pts_left,
                    pts_right=pts_right,
                    pts_mouth=pts_mouth,
                )

                # Visual Feedback: Dynamic Eye Contours
                if pts_left and ear_left is not None:
                    color = (
                        Config.COLOR_DROWSY
                        if ear_left < self.detector.ear_threshold
                        else Config.COLOR_EYE_CONTOUR
                    )
                    HUDOverlay.draw_contours(frame, pts_left, self.cfg.LEFT_EYE_CONTOUR, color)

                if pts_right and ear_right is not None:
                    color = (
                        Config.COLOR_DROWSY
                        if ear_right < self.detector.ear_threshold
                        else Config.COLOR_EYE_CONTOUR
                    )
                    HUDOverlay.draw_contours(frame, pts_right, self.cfg.RIGHT_EYE_CONTOUR, color)

                # Visual Feedback: Dynamic Mouth Contour
                if pts_mouth and self.detector.enable_yawn:
                    mouth_color = (
                        Config.COLOR_WARNING
                        if (mar is not None and mar >= self.detector.mar_threshold)
                        else Config.COLOR_MOUTH_CONTOUR
                    )
                    HUDOverlay.draw_contours(
                        frame, pts_mouth, self.cfg.MOUTH_CONTOUR, mouth_color, point_color=(0, 200, 255)
                    )

                # Smooth FPS calculation
                curr_time = time.time()
                time_delta = curr_time - prev_time
                prev_time = curr_time
                if time_delta > 0:
                    current_fps = 1.0 / time_delta
                    fps = (0.90 * fps) + (0.10 * current_fps)

                # Render HUD
                HUDOverlay.render_hud(
                    frame=frame,
                    status=status,
                    fps=fps,
                    detector=self.detector,
                    calibration_progress=calibration_progress,
                )

                # Display Canvas
                cv2.imshow(self.cfg.WINDOW_TITLE, frame)

                # Keyboard Controls
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    logger.info("Shutdown signal received from user.")
                    break
                elif key in (ord("+"), ord("=")):  # Increase sensitivity
                    self.detector.ear_threshold = round(
                        min(0.40, self.detector.ear_threshold + 0.01), 2
                    )
                    logger.info(f"Adjusted EAR Threshold -> {self.detector.ear_threshold}")
                elif key in (ord("-"), ord("_")):  # Decrease sensitivity
                    self.detector.ear_threshold = round(
                        max(0.12, self.detector.ear_threshold - 0.01), 2
                    )
                    logger.info(f"Adjusted EAR Threshold -> {self.detector.ear_threshold}")
                elif key in (ord("c"), ord("C")):  # Trigger auto-calibration
                    self.trigger_calibration()
                elif key in (ord("m"), ord("M")):  # Toggle Yawn Detection
                    self.detector.enable_yawn = not self.detector.enable_yawn
                    logger.info(f"Yawn Detection Toggled -> {'ENABLED' if self.detector.enable_yawn else 'DISABLED'}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user (KeyboardInterrupt).")
        except Exception as exc:
            logger.exception(f"Unexpected runtime error: {exc}")
        finally:
            self._cleanup(face_mesh)

    def _cleanup(self, face_mesh) -> None:
        """Gracefully release camera, audio engine, and OpenCV windows."""
        logger.info("Initiating graceful system shutdown...")
        try:
            face_mesh.close()
        except Exception:
            pass

        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            logger.info("Camera stream released.")

        self.audio_engine.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown completed cleanly.")


# ============================================================================
# CLI ARGUMENT PARSER & ENTRY POINT
# ============================================================================
def parse_arguments() -> Config:
    """Parse command-line arguments to construct Config."""
    parser = argparse.ArgumentParser(
        description="Real-time Driver / User Drowsiness & Fatigue Detection System."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to an optional video file (e.g. dashcam footage).",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=Config.EAR_THRESHOLD,
        help=f"Eye Aspect Ratio threshold (default: {Config.EAR_THRESHOLD}).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=Config.CLOSED_DURATION_THRESHOLD,
        help=f"Eye closure duration in seconds to trigger alarm (default: {Config.CLOSED_DURATION_THRESHOLD}s).",
    )
    parser.add_argument(
        "--alarm-audio",
        type=str,
        default=Config.ALARM_AUDIO_PATH,
        help=f"Path to alarm sound file (default: '{Config.ALARM_AUDIO_PATH}').",
    )
    parser.add_argument(
        "--no-yawn",
        action="store_true",
        help="Disable yawn (MAR) fatigue detection.",
    )

    args = parser.parse_args()

    cfg = Config(
        CAMERA_INDEX=args.camera,
        VIDEO_PATH=args.video,
        EAR_THRESHOLD=args.ear_threshold,
        CLOSED_DURATION_THRESHOLD=args.duration,
        ALARM_AUDIO_PATH=args.alarm_audio,
        ENABLE_YAWN_DETECTION=not args.no_yawn,
    )
    return cfg


def main() -> None:
    """Script entry point."""
    cfg = parse_arguments()
    app = DrowsinessDetectorApp(config=cfg)
    app.run()


if __name__ == "__main__":
    main()
