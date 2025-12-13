---
id: ch15-multimodal-perception
title: "Chapter 15: Multimodal Perception for Humanoids"
sidebar_position: 4
---

# Chapter 15: Multimodal Perception for Humanoids

**Estimated Time**: 5-6 hours | **Exercises**: 4

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Integrate** multiple sensor modalities for robust perception
2. **Implement** visual-language grounding for object detection
3. **Process** audio for speech and sound recognition
4. **Fuse** tactile feedback with visual information
5. **Build** a unified perception pipeline for humanoid robots

---

## 15.1 Multimodal Perception Overview

Humanoid robots require multiple senses working together for effective interaction.

### Sensor Modalities

```
┌─────────────────────────────────────────────────────────┐
│            Humanoid Perception System                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Vision              Audio              Tactile         │
│   ──────              ─────              ───────         │
│   • RGB cameras       • Microphones      • Force/torque  │
│   • Depth sensors     • Arrays           • Skin sensors  │
│   • Stereo vision     • Directional      • Pressure      │
│                                                          │
│   Proprioception      Proximity          Other           │
│   ──────────────      ─────────          ─────           │
│   • Joint encoders    • Ultrasonic       • Temperature   │
│   • IMU               • Infrared         • Humidity      │
│   • Force sensors     • Capacitive       • Gas sensors   │
│                                                          │
│                    ↓                                     │
│            ┌─────────────────┐                          │
│            │  Fusion Module  │                          │
│            └────────┬────────┘                          │
│                     ↓                                    │
│            Unified World Model                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Perception Architecture

```python
# perception_pipeline.py
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np

@dataclass
class SensorData:
    """Container for multimodal sensor data."""
    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    audio: Optional[np.ndarray] = None
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    force_torque: Optional[np.ndarray] = None
    tactile: Optional[np.ndarray] = None
    timestamp: float = 0.0

@dataclass
class PerceptionOutput:
    """Output of perception pipeline."""
    detected_objects: List[Dict]
    scene_description: str
    speech_text: Optional[str]
    contact_info: Dict
    pose_estimate: np.ndarray
    confidence: float

class MultimodalPerceptionPipeline:
    """Unified multimodal perception for humanoids."""

    def __init__(self, config: Dict):
        self.config = config

        # Initialize encoders
        self.vision_encoder = VisionEncoder(config.get("vision", {}))
        self.audio_encoder = AudioEncoder(config.get("audio", {}))
        self.tactile_encoder = TactileEncoder(config.get("tactile", {}))
        self.proprioception_encoder = ProprioceptionEncoder(
            config.get("proprioception", {})
        )

        # Fusion module
        self.fusion = MultimodalFusion(config.get("fusion", {}))

        # Task-specific heads
        self.object_detector = ObjectDetectionHead()
        self.speech_recognizer = SpeechRecognitionHead()
        self.scene_understander = SceneUnderstandingHead()

    def process(self, sensor_data: SensorData) -> PerceptionOutput:
        """Process all sensor modalities."""
        features = {}

        # Encode each modality
        if sensor_data.rgb_image is not None:
            features["vision"] = self.vision_encoder(
                sensor_data.rgb_image,
                sensor_data.depth_image
            )

        if sensor_data.audio is not None:
            features["audio"] = self.audio_encoder(sensor_data.audio)

        if sensor_data.tactile is not None:
            features["tactile"] = self.tactile_encoder(sensor_data.tactile)

        if sensor_data.joint_positions is not None:
            features["proprioception"] = self.proprioception_encoder(
                sensor_data.joint_positions,
                sensor_data.joint_velocities,
                sensor_data.force_torque
            )

        # Fuse modalities
        fused_features = self.fusion(features)

        # Generate outputs
        detected_objects = self.object_detector(fused_features, features.get("vision"))
        speech_text = self.speech_recognizer(features.get("audio"))
        scene_description = self.scene_understander(fused_features)

        # Contact information from tactile
        contact_info = self._process_contact(
            features.get("tactile"),
            features.get("proprioception")
        )

        return PerceptionOutput(
            detected_objects=detected_objects,
            scene_description=scene_description,
            speech_text=speech_text,
            contact_info=contact_info,
            pose_estimate=self._estimate_pose(features.get("proprioception")),
            confidence=self._compute_confidence(features)
        )

    def _process_contact(self, tactile_features, proprio_features) -> Dict:
        """Process contact information."""
        return {
            "in_contact": False,  # Placeholder
            "contact_force": np.zeros(3),
            "contact_location": None
        }

    def _estimate_pose(self, proprio_features) -> np.ndarray:
        """Estimate robot pose."""
        return np.zeros(7)  # Placeholder

    def _compute_confidence(self, features: Dict) -> float:
        """Compute overall perception confidence."""
        return 0.9  # Placeholder
```

---

## 15.2 Visual-Language Grounding

Ground natural language descriptions in visual observations.

### Open-Vocabulary Detection

```python
# visual_language_grounding.py
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Detection:
    """Single object detection."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]
    mask: Optional[np.ndarray] = None

class OpenVocabularyDetector:
    """Open-vocabulary object detection using OWL-ViT."""

    def __init__(self, model_name: str = "google/owlvit-base-patch32"):
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def detect(
        self,
        image: np.ndarray,
        text_queries: List[str],
        threshold: float = 0.1
    ) -> List[Detection]:
        """
        Detect objects matching text queries.

        Args:
            image: RGB image as numpy array
            text_queries: List of object descriptions
            threshold: Detection confidence threshold

        Returns:
            List of Detection objects
        """
        # Convert to PIL
        pil_image = Image.fromarray(image)

        # Process inputs
        inputs = self.processor(
            text=text_queries,
            images=pil_image,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Detect
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor([pil_image.size[::-1]])
        if torch.cuda.is_available():
            target_sizes = target_sizes.cuda()

        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]

        detections = []
        for score, label_idx, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            x1, y1, x2, y2 = box.int().tolist()
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            detections.append(Detection(
                label=text_queries[label_idx],
                confidence=score.item(),
                bbox=(x1, y1, x2, y2),
                center=center
            ))

        return detections

    def detect_with_prompt(
        self,
        image: np.ndarray,
        natural_prompt: str
    ) -> List[Detection]:
        """
        Detect objects from natural language prompt.

        Example: "Find the red cup on the table"
        """
        # Extract object queries from prompt
        queries = self._extract_queries(natural_prompt)
        return self.detect(image, queries)

    def _extract_queries(self, prompt: str) -> List[str]:
        """Extract object queries from natural prompt."""
        # Simple extraction - could use LLM for better parsing
        common_objects = [
            "cup", "bottle", "box", "ball", "book", "phone",
            "remote", "pen", "scissors", "mug", "plate", "bowl"
        ]

        queries = []
        prompt_lower = prompt.lower()

        for obj in common_objects:
            if obj in prompt_lower:
                queries.append(obj)

        # Add color variants
        colors = ["red", "blue", "green", "yellow", "white", "black"]
        enhanced_queries = []
        for query in queries:
            enhanced_queries.append(query)
            for color in colors:
                if color in prompt_lower:
                    enhanced_queries.append(f"{color} {query}")

        return enhanced_queries if enhanced_queries else ["object"]


class GroundedSegmentation:
    """Combine detection with segmentation."""

    def __init__(self):
        # Using SAM for segmentation
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
        self.predictor = SamPredictor(sam)

        self.detector = OpenVocabularyDetector()

    def segment_by_text(
        self,
        image: np.ndarray,
        text_query: str
    ) -> List[Detection]:
        """Detect and segment objects by text description."""
        # First detect
        detections = self.detector.detect(image, [text_query])

        # Then segment each detection
        self.predictor.set_image(image)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            input_box = np.array([x1, y1, x2, y2])

            masks, scores, _ = self.predictor.predict(
                box=input_box,
                multimask_output=False
            )

            det.mask = masks[0]

        return detections
```

---

## 15.3 Audio Processing

### Speech Recognition and Sound Understanding

```python
# audio_processing.py
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Optional, Dict, List
import librosa

class SpeechRecognizer:
    """Speech-to-text using Whisper."""

    def __init__(self, model_name: str = "openai/whisper-base"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.sample_rate = 16000

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """
        Transcribe speech to text.

        Args:
            audio: Audio waveform
            sample_rate: Audio sample rate

        Returns:
            Transcribed text
        """
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate
            )

        # Process
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(inputs["input_features"])

        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return transcription.strip()


class SoundClassifier:
    """Classify environmental sounds."""

    def __init__(self):
        # Using a pre-trained audio classifier
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.model.eval()

    def classify(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Classify audio into categories.

        Returns:
            List of {label, score} dicts
        """
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, top_k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                "label": self.model.config.id2label[idx.item()],
                "score": prob.item()
            })

        return results


class SpeakerLocalization:
    """Localize sound sources using microphone array."""

    def __init__(self, mic_positions: np.ndarray):
        """
        Args:
            mic_positions: [N, 3] array of microphone positions
        """
        self.mic_positions = mic_positions
        self.num_mics = len(mic_positions)
        self.speed_of_sound = 343.0  # m/s

    def localize(
        self,
        audio_channels: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Estimate direction of arrival using GCC-PHAT.

        Args:
            audio_channels: [N, T] multi-channel audio
            sample_rate: Sample rate

        Returns:
            Estimated source direction [azimuth, elevation]
        """
        # Cross-correlation between channel pairs
        delays = []

        for i in range(self.num_mics):
            for j in range(i + 1, self.num_mics):
                delay = self._gcc_phat(
                    audio_channels[i],
                    audio_channels[j],
                    sample_rate
                )
                delays.append((i, j, delay))

        # Estimate direction using TDOA
        direction = self._tdoa_localization(delays)

        return direction

    def _gcc_phat(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray,
        sample_rate: int
    ) -> float:
        """Generalized Cross-Correlation with Phase Transform."""
        n = len(sig1) + len(sig2)

        # FFT
        SIG1 = np.fft.fft(sig1, n=n)
        SIG2 = np.fft.fft(sig2, n=n)

        # Cross-spectrum
        R = SIG1 * np.conj(SIG2)

        # PHAT weighting
        R /= np.abs(R) + 1e-10

        # Inverse FFT
        cc = np.fft.ifft(R).real

        # Find peak
        max_idx = np.argmax(np.abs(cc))
        if max_idx > n // 2:
            max_idx -= n

        delay = max_idx / sample_rate
        return delay

    def _tdoa_localization(self, delays: List) -> np.ndarray:
        """Localize using time difference of arrival."""
        # Simplified - actual implementation needs optimization
        return np.array([0.0, 0.0])  # azimuth, elevation


class AudioPerceptionModule:
    """Combined audio perception for humanoids."""

    def __init__(self, mic_positions: np.ndarray):
        self.speech_recognizer = SpeechRecognizer()
        self.sound_classifier = SoundClassifier()
        self.speaker_localizer = SpeakerLocalization(mic_positions)

        self.speech_threshold = 0.5

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict:
        """Process audio for speech and sounds."""
        results = {
            "speech_text": None,
            "sound_classes": [],
            "speaker_direction": None
        }

        # Classify sound
        sound_classes = self.sound_classifier.classify(audio, sample_rate)
        results["sound_classes"] = sound_classes

        # Check for speech
        has_speech = any(
            "speech" in c["label"].lower() and c["score"] > self.speech_threshold
            for c in sound_classes
        )

        if has_speech:
            results["speech_text"] = self.speech_recognizer.transcribe(
                audio, sample_rate
            )

        # Localize (if multi-channel)
        if audio.ndim == 2:
            results["speaker_direction"] = self.speaker_localizer.localize(
                audio, sample_rate
            )

        return results
```

---

## 15.4 Tactile Perception

### Force and Contact Processing

```python
# tactile_perception.py
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.signal import butter, filtfilt

@dataclass
class ContactEvent:
    """Detected contact event."""
    timestamp: float
    location: np.ndarray  # Position on robot
    force: np.ndarray  # Force vector
    torque: np.ndarray  # Torque vector
    contact_type: str  # "impact", "sustained", "slip"

class TactileSensor:
    """Base class for tactile sensors."""

    def __init__(self, position: np.ndarray, orientation: np.ndarray):
        self.position = position
        self.orientation = orientation

    def read(self) -> np.ndarray:
        """Read sensor values."""
        raise NotImplementedError

class ForceTorqueSensor(TactileSensor):
    """6-axis force/torque sensor."""

    def __init__(self, position, orientation, interface):
        super().__init__(position, orientation)
        self.interface = interface

        # Calibration
        self.bias = np.zeros(6)
        self.scale = np.ones(6)

        # Filter
        self.filter_order = 2
        self.cutoff_freq = 10  # Hz
        self.sample_rate = 1000  # Hz

    def read(self) -> np.ndarray:
        """Read force/torque values."""
        raw = self.interface.read()

        # Apply calibration
        calibrated = (raw - self.bias) * self.scale

        return calibrated

    def get_force(self) -> np.ndarray:
        """Get force vector [Fx, Fy, Fz]."""
        return self.read()[:3]

    def get_torque(self) -> np.ndarray:
        """Get torque vector [Tx, Ty, Tz]."""
        return self.read()[3:]

    def calibrate(self, num_samples: int = 100):
        """Calibrate sensor bias."""
        samples = []
        for _ in range(num_samples):
            samples.append(self.interface.read())

        self.bias = np.mean(samples, axis=0)


class TactileSkin:
    """Distributed tactile skin sensor."""

    def __init__(
        self,
        num_taxels: int,
        taxel_positions: np.ndarray,
        interface
    ):
        self.num_taxels = num_taxels
        self.taxel_positions = taxel_positions  # [N, 3]
        self.interface = interface

        # Contact detection threshold
        self.contact_threshold = 0.1

    def read(self) -> np.ndarray:
        """Read all taxel values."""
        return self.interface.read()

    def get_contacts(self) -> List[Tuple[np.ndarray, float]]:
        """Get list of contact locations and pressures."""
        values = self.read()
        contacts = []

        for i, val in enumerate(values):
            if val > self.contact_threshold:
                contacts.append((self.taxel_positions[i], val))

        return contacts

    def get_contact_centroid(self) -> Optional[np.ndarray]:
        """Get pressure-weighted contact centroid."""
        contacts = self.get_contacts()

        if not contacts:
            return None

        positions = np.array([c[0] for c in contacts])
        pressures = np.array([c[1] for c in contacts])

        centroid = np.average(positions, axis=0, weights=pressures)
        return centroid


class ContactDetector:
    """Detect and classify contact events."""

    def __init__(self, ft_sensor: ForceTorqueSensor):
        self.sensor = ft_sensor

        # Detection parameters
        self.impact_threshold = 10.0  # N
        self.slip_threshold = 2.0  # N
        self.sustained_time = 0.1  # seconds

        # State
        self.in_contact = False
        self.contact_start_time = None
        self.force_history = []
        self.max_history = 100

    def update(self, timestamp: float) -> Optional[ContactEvent]:
        """Update detector and return contact event if detected."""
        force = self.sensor.get_force()
        torque = self.sensor.get_torque()
        force_magnitude = np.linalg.norm(force)

        # Store history
        self.force_history.append((timestamp, force))
        if len(self.force_history) > self.max_history:
            self.force_history.pop(0)

        event = None

        if not self.in_contact:
            # Check for new contact
            if force_magnitude > self.impact_threshold:
                self.in_contact = True
                self.contact_start_time = timestamp

                event = ContactEvent(
                    timestamp=timestamp,
                    location=self.sensor.position,
                    force=force,
                    torque=torque,
                    contact_type="impact"
                )

        else:
            # Already in contact
            if force_magnitude < self.impact_threshold * 0.5:
                # Contact ended
                self.in_contact = False

            else:
                # Check for slip
                if self._detect_slip():
                    event = ContactEvent(
                        timestamp=timestamp,
                        location=self.sensor.position,
                        force=force,
                        torque=torque,
                        contact_type="slip"
                    )

        return event

    def _detect_slip(self) -> bool:
        """Detect slip from force history."""
        if len(self.force_history) < 10:
            return False

        recent = np.array([f[1] for f in self.force_history[-10:]])

        # Check for sudden tangential force changes
        tangential = recent[:, :2]  # Assuming Z is normal
        tangential_var = np.var(tangential, axis=0).sum()

        return tangential_var > self.slip_threshold ** 2


class GraspStabilityMonitor:
    """Monitor grasp stability using tactile feedback."""

    def __init__(self, finger_sensors: List[ForceTorqueSensor]):
        self.sensors = finger_sensors
        self.contact_detectors = [
            ContactDetector(s) for s in finger_sensors
        ]

        # Stability criteria
        self.min_contacts = 2
        self.force_balance_threshold = 5.0

    def is_stable(self) -> Tuple[bool, Dict]:
        """Check if current grasp is stable."""
        forces = [s.get_force() for s in self.sensors]
        contacts = [np.linalg.norm(f) > 1.0 for f in forces]

        # Check minimum contacts
        num_contacts = sum(contacts)
        if num_contacts < self.min_contacts:
            return False, {"reason": "insufficient_contacts", "num_contacts": num_contacts}

        # Check force balance
        total_force = np.sum(forces, axis=0)
        force_imbalance = np.linalg.norm(total_force)

        if force_imbalance > self.force_balance_threshold:
            return False, {"reason": "force_imbalance", "imbalance": force_imbalance}

        return True, {"num_contacts": num_contacts, "force_imbalance": force_imbalance}

    def predict_slip(self) -> float:
        """Predict probability of slip."""
        # Simple heuristic based on force ratios
        forces = [s.get_force() for s in self.sensors]

        slip_probability = 0.0
        for force in forces:
            normal = abs(force[2])
            tangential = np.linalg.norm(force[:2])

            if normal > 0.1:
                friction_ratio = tangential / normal
                # Assuming friction coefficient ~0.5
                slip_probability = max(slip_probability, friction_ratio / 0.5)

        return min(1.0, slip_probability)
```

---

## 15.5 Sensor Fusion

### Multimodal Fusion Module

```python
# sensor_fusion.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional

class MultimodalFusion(nn.Module):
    """Fuse features from multiple modalities."""

    def __init__(self, config: Dict):
        super().__init__()

        self.modality_dims = config.get("modality_dims", {
            "vision": 768,
            "audio": 512,
            "tactile": 128,
            "proprioception": 256
        })

        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_heads = config.get("num_heads", 8)

        # Project each modality to common dimension
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, self.hidden_dim)
            for name, dim in self.modality_dims.items()
        })

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True
        )

        # Self-attention for fused features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Modality importance weights
        self.modality_weights = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1))
            for name in self.modality_dims.keys()
        })

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multimodal features.

        Args:
            features: Dict of {modality_name: tensor}

        Returns:
            Fused feature tensor
        """
        # Project all modalities
        projected = {}
        for name, feat in features.items():
            if name in self.projections:
                proj = self.projections[name](feat)
                weight = torch.sigmoid(self.modality_weights[name])
                projected[name] = proj * weight

        if not projected:
            raise ValueError("No valid modalities provided")

        # Concatenate along sequence dimension
        # Each modality contributes tokens
        all_tokens = []
        for name, feat in projected.items():
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)  # Add sequence dim
            all_tokens.append(feat)

        concat_features = torch.cat(all_tokens, dim=1)

        # Self-attention
        attended, _ = self.self_attention(
            concat_features,
            concat_features,
            concat_features
        )

        # Global pooling
        pooled = attended.mean(dim=1)

        # Output projection
        output = self.output_proj(pooled)

        return output


class TemporalFusion(nn.Module):
    """Fuse features across time."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

    def forward(
        self,
        features: torch.Tensor,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, D] temporal features
            return_sequence: Return full sequence or just final

        Returns:
            Fused features
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(features)

        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)

        if return_sequence:
            return attended
        else:
            # Return last timestep
            return attended[:, -1, :]


class UncertaintyFusion(nn.Module):
    """Fuse modalities with uncertainty estimation."""

    def __init__(self, modality_dims: Dict[str, int], output_dim: int):
        super().__init__()

        self.modality_dims = modality_dims

        # Each modality predicts mean and variance
        self.modality_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, output_dim * 2)  # mean + log_var
            )
            for name, dim in modality_dims.items()
        })

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse with uncertainty weighting.

        More certain modalities contribute more.
        """
        means = []
        precisions = []  # 1/variance

        for name, feat in features.items():
            if name not in self.modality_heads:
                continue

            output = self.modality_heads[name](feat)
            mean, log_var = output.chunk(2, dim=-1)

            # Precision = 1/variance
            precision = torch.exp(-log_var)

            means.append(mean)
            precisions.append(precision)

        if not means:
            raise ValueError("No valid modalities")

        # Stack
        means = torch.stack(means, dim=0)  # [M, B, D]
        precisions = torch.stack(precisions, dim=0)

        # Precision-weighted average
        total_precision = precisions.sum(dim=0)
        weighted_mean = (means * precisions).sum(dim=0) / (total_precision + 1e-6)

        return weighted_mean
```

---

## Exercises

### Exercise 15.1: Visual-Language Detection

**Objective**: Detect objects from text descriptions.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Load OWL-ViT model
2. Test with robot camera images
3. Compare different text prompts
4. Measure detection accuracy

---

### Exercise 15.2: Speech Recognition System

**Objective**: Add speech understanding to robot.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Set up Whisper model
2. Process microphone audio
3. Handle streaming input
4. Test with robot commands

---

### Exercise 15.3: Contact Detection

**Objective**: Detect and classify contacts.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Implement force threshold detection
2. Add slip detection
3. Test with manipulation tasks
4. Visualize contact events

---

### Exercise 15.4: Sensor Fusion Pipeline

**Objective**: Build complete perception pipeline.

**Difficulty**: Advanced | **Estimated Time**: 90 minutes

#### Instructions

1. Combine vision, audio, tactile
2. Implement feature fusion
3. Add temporal integration
4. Test end-to-end system

---

## Summary

In this chapter, you learned:

- **Multimodal perception** combines vision, audio, and touch
- **Visual-language grounding** enables object detection from text
- **Audio processing** provides speech and sound understanding
- **Tactile perception** enables safe manipulation
- **Sensor fusion** creates unified world understanding

---

## References

[1] A. Kirillov et al., "Segment Anything," in *ICCV*, 2023.

[2] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," in *ICML*, 2023.

[3] R. Calandra et al., "The Feeling of Success: Does Touch Sensing Help Predict Grasp Outcomes?," in *CoRL*, 2017.

[4] Y. Lee et al., "Multimodal Sensor Fusion with Transformers for Robot Manipulation," in *ICRA*, 2023.
