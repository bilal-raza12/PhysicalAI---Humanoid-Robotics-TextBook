---
id: ch10-synthetic-data
title: "Chapter 10: Synthetic Data Generation & Domain Randomization"
sidebar_position: 3
---

# Chapter 10: Synthetic Data Generation & Domain Randomization

**Estimated Time**: 5-6 hours | **Exercises**: 4

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Generate** synthetic training data using Isaac Sim Replicator
2. **Apply** domain randomization for robust perception
3. **Create** diverse training scenarios automatically
4. **Export** data in standard formats (COCO, KITTI)
5. **Validate** synthetic data quality for training

---

## 10.1 Introduction to Synthetic Data

Synthetic data addresses the challenge of obtaining labeled training data for robotics.

### Benefits of Synthetic Data

| Advantage | Description |
|-----------|-------------|
| Unlimited quantity | Generate millions of samples |
| Perfect labels | Automatic ground truth |
| Rare scenarios | Create edge cases easily |
| Privacy-safe | No real human data needed |
| Cost-effective | No manual annotation |

### Isaac Sim Replicator

Replicator is Isaac Sim's synthetic data generation framework.

```python
# replicator_basics.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep

# Set output directory
rep.settings.carb_settings("/omni/replicator/RTSubframes", 1)

# Create simple scene
with rep.new_layer():
    # Camera
    camera = rep.create.camera(
        position=(3, 3, 3),
        look_at=(0, 0, 0)
    )

    # Light
    rep.create.light(
        light_type="dome",
        intensity=1000
    )

    # Objects
    cube = rep.create.cube(
        semantics=[("class", "object")],
        position=(0, 0, 0.5)
    )

# Render product
render_product = rep.create.render_product(camera, (640, 480))

# Writers for different outputs
rgb_writer = rep.WriterRegistry.get("BasicWriter")
rgb_writer.initialize(
    output_dir="_output/rgb",
    rgb=True
)
rgb_writer.attach([render_product])

# Run generation
rep.orchestrator.run()

simulation_app.close()
```

---

## 10.2 Domain Randomization

Domain randomization improves sim-to-real transfer by varying simulation parameters.

### Types of Randomization

```
┌─────────────────────────────────────────────────────────┐
│                Domain Randomization                      │
├─────────────────┬───────────────────┬───────────────────┤
│   Appearance    │    Geometry       │    Physics        │
├─────────────────┼───────────────────┼───────────────────┤
│ - Textures      │ - Object scale    │ - Friction        │
│ - Lighting      │ - Object pose     │ - Mass            │
│ - Colors        │ - Camera position │ - Joint damping   │
│ - Materials     │ - Distractors     │ - Motor strength  │
└─────────────────┴───────────────────┴───────────────────┘
```

### Implementing Domain Randomization

```python
# domain_randomization.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import numpy as np

with rep.new_layer():
    # Create camera with randomized position
    camera = rep.create.camera()

    # Randomize camera each frame
    with rep.trigger.on_frame(num_frames=100):
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (2, -2, 1.5),
                    (4, 2, 3)
                ),
                look_at=(0, 0, 0.5)
            )

    # Create target object
    target = rep.create.cube(
        semantics=[("class", "target")],
        position=(0, 0, 0.5)
    )

    # Randomize target appearance
    with rep.trigger.on_frame():
        with target:
            rep.randomizer.materials(
                materials=rep.create.material_omnipbr(
                    diffuse=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
                    roughness=rep.distribution.uniform(0.1, 0.9),
                    metallic=rep.distribution.uniform(0, 1)
                )
            )
            rep.modify.pose(
                rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360))
            )

    # Randomize lighting
    light = rep.create.light(
        light_type="sphere",
        intensity=rep.distribution.uniform(500, 2000),
        position=rep.distribution.uniform((-3, -3, 2), (3, 3, 5))
    )

    # Add distractor objects
    def random_distractors():
        distractors = rep.create.cube(
            count=rep.distribution.uniform(3, 10),
            semantics=[("class", "distractor")],
            position=rep.distribution.uniform((-2, -2, 0), (2, 2, 1)),
            scale=rep.distribution.uniform(0.1, 0.3)
        )
        with distractors:
            rep.randomizer.materials(
                materials=rep.create.material_omnipbr(
                    diffuse=rep.distribution.uniform((0, 0, 0), (1, 1, 1))
                )
            )
        return distractors

    rep.randomizer.register(random_distractors)

    with rep.trigger.on_frame():
        rep.randomizer.random_distractors()

# Setup render
render_product = rep.create.render_product(camera, (640, 480))

# Output annotations
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="_output/domain_rand",
    rgb=True,
    bounding_box_2d_tight=True,
    semantic_segmentation=True,
    instance_segmentation=True
)
writer.attach([render_product])

rep.orchestrator.run()
simulation_app.close()
```

---

## 10.3 Annotation Types

### Available Annotations

| Type | Description | Use Case |
|------|-------------|----------|
| RGB | Color images | Visual perception |
| Depth | Distance per pixel | 3D reconstruction |
| Semantic Seg | Class per pixel | Scene understanding |
| Instance Seg | Object per pixel | Object detection |
| 2D BBox | Object rectangles | Detection training |
| 3D BBox | Object cuboids | 3D detection |
| Keypoints | Joint positions | Pose estimation |
| Normals | Surface direction | Surface reconstruction |

### Multi-Annotation Setup

```python
# multi_annotation.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep

with rep.new_layer():
    # Scene setup
    camera = rep.create.camera(position=(3, 0, 2), look_at=(0, 0, 0.5))
    rep.create.light(light_type="dome", intensity=1000)

    # Robot with semantic labels
    robot = rep.create.from_usd(
        "/path/to/humanoid.usd",
        semantics=[
            ("class", "robot"),
            ("part", "humanoid")
        ]
    )

    # Add keypoint annotations for joints
    # Define skeleton structure
    keypoint_config = {
        "head": "/humanoid/head",
        "left_shoulder": "/humanoid/left_arm/shoulder",
        "left_elbow": "/humanoid/left_arm/elbow",
        "left_wrist": "/humanoid/left_arm/wrist",
        "right_shoulder": "/humanoid/right_arm/shoulder",
        "right_elbow": "/humanoid/right_arm/elbow",
        "right_wrist": "/humanoid/right_arm/wrist",
        "pelvis": "/humanoid/pelvis",
        "left_hip": "/humanoid/left_leg/hip",
        "left_knee": "/humanoid/left_leg/knee",
        "left_ankle": "/humanoid/left_leg/ankle",
        "right_hip": "/humanoid/right_leg/hip",
        "right_knee": "/humanoid/right_leg/knee",
        "right_ankle": "/humanoid/right_leg/ankle",
    }

# Render products
render_product = rep.create.render_product(camera, (1280, 720))

# Comprehensive writer
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="_output/annotations",
    rgb=True,
    distance_to_camera=True,
    bounding_box_2d_tight=True,
    bounding_box_2d_loose=True,
    bounding_box_3d=True,
    semantic_segmentation=True,
    instance_segmentation=True,
    normals=True,
    motion_vectors=True,
    occlusion=True
)
writer.attach([render_product])

# Generate frames
rep.orchestrator.run()
simulation_app.close()
```

---

## 10.4 Object Detection Dataset

### COCO Format Export

```python
# coco_dataset.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import json
import os

# Output configuration
OUTPUT_DIR = "_output/coco_dataset"
NUM_FRAMES = 1000

with rep.new_layer():
    # Environment
    rep.create.light(light_type="dome", intensity=1000)

    # Ground
    ground = rep.create.plane(
        scale=10,
        semantics=[("class", "background")]
    )

    # Camera randomization
    camera = rep.create.camera()
    with rep.trigger.on_frame(num_frames=NUM_FRAMES):
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform((2, -2, 1), (4, 2, 3)),
                look_at=(0, 0, 0.5)
            )

    # Objects to detect
    objects_config = [
        {"class": "cup", "usd": "/assets/cup.usd"},
        {"class": "bottle", "usd": "/assets/bottle.usd"},
        {"class": "box", "usd": "/assets/box.usd"},
    ]

    for obj_cfg in objects_config:
        obj = rep.create.from_usd(
            obj_cfg["usd"],
            semantics=[("class", obj_cfg["class"])]
        )
        with rep.trigger.on_frame():
            with obj:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (-1, -1, 0),
                        (1, 1, 0.5)
                    ),
                    rotation=rep.distribution.uniform(
                        (0, 0, 0),
                        (0, 0, 360)
                    )
                )

# Render setup
render_product = rep.create.render_product(camera, (640, 480))

# COCO writer
coco_writer = rep.WriterRegistry.get("CocoWriter")
coco_writer.initialize(
    output_dir=OUTPUT_DIR,
    semantic_types=["class"],
    image_output_format="png"
)
coco_writer.attach([render_product])

rep.orchestrator.run()
simulation_app.close()

print(f"COCO dataset generated at {OUTPUT_DIR}")
```

### Custom Dataset Writer

```python
# custom_writer.py
import omni.replicator.core as rep
from omni.replicator.core import Writer, AnnotatorRegistry
import numpy as np
import json
import os

class HumanoidPoseWriter(Writer):
    """Custom writer for humanoid pose estimation datasets."""

    def __init__(
        self,
        output_dir: str,
        image_format: str = "png",
        skeleton_config: dict = None
    ):
        self.output_dir = output_dir
        self.image_format = image_format
        self.skeleton_config = skeleton_config or {}

        self.frame_id = 0
        self.annotations = []

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

        # Backend setup
        self.backend = rep.backends.get("disk")
        self.backend.initialize(output_dir=output_dir)

    def write(self, data: dict):
        """Write single frame data."""
        # Save RGB image
        rgb = data.get("rgb")
        if rgb is not None:
            image_path = f"images/frame_{self.frame_id:06d}.{self.image_format}"
            self.backend.write_image(image_path, rgb)

        # Extract keypoints
        keypoints_3d = data.get("keypoints_3d", {})
        keypoints_2d = data.get("keypoints_2d", {})

        annotation = {
            "frame_id": self.frame_id,
            "image_file": image_path,
            "keypoints_3d": keypoints_3d,
            "keypoints_2d": keypoints_2d,
            "bbox": data.get("bounding_box_2d_tight", [])
        }

        self.annotations.append(annotation)
        self.frame_id += 1

    def on_final_frame(self):
        """Save final annotations file."""
        output_file = os.path.join(self.output_dir, "annotations.json")

        dataset = {
            "info": {
                "description": "Humanoid Pose Dataset",
                "version": "1.0",
                "skeleton": self.skeleton_config
            },
            "annotations": self.annotations
        }

        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"Saved {len(self.annotations)} annotations to {output_file}")

# Register custom writer
rep.WriterRegistry.register(HumanoidPoseWriter)
```

---

## 10.5 Scenario Generation

### Humanoid Interaction Scenarios

```python
# interaction_scenarios.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import random

class ScenarioGenerator:
    """Generate diverse HRI scenarios."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.scenarios = [
            "handover",
            "navigation",
            "manipulation",
            "social_interaction"
        ]

    def create_handover_scenario(self):
        """Object handover between human and robot."""
        with rep.new_layer():
            # Human model
            human = rep.create.from_usd(
                "/assets/human.usd",
                semantics=[("class", "human"), ("role", "partner")]
            )

            # Robot
            robot = rep.create.from_usd(
                "/assets/humanoid_robot.usd",
                semantics=[("class", "robot")]
            )

            # Handover object
            obj = rep.create.from_usd(
                "/assets/cup.usd",
                semantics=[("class", "object"), ("task", "handover")]
            )

            # Position for handover
            with rep.trigger.on_frame():
                with human:
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (0.8, -0.3, 0),
                            (1.2, 0.3, 0)
                        )
                    )
                with robot:
                    rep.modify.pose(position=(0, 0, 0))
                with obj:
                    # Object between human and robot
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (0.3, -0.1, 0.8),
                            (0.6, 0.1, 1.2)
                        )
                    )

    def create_navigation_scenario(self):
        """Robot navigating among humans."""
        with rep.new_layer():
            # Robot
            robot = rep.create.from_usd(
                "/assets/humanoid_robot.usd",
                semantics=[("class", "robot")]
            )

            # Crowd of humans
            for i in range(5):
                human = rep.create.from_usd(
                    "/assets/human.usd",
                    semantics=[("class", "human"), ("id", str(i))]
                )
                with rep.trigger.on_frame():
                    with human:
                        rep.modify.pose(
                            position=rep.distribution.uniform(
                                (-3, -3, 0),
                                (3, 3, 0)
                            ),
                            rotation=rep.distribution.uniform(
                                (0, 0, 0),
                                (0, 0, 360)
                            )
                        )

            # Obstacles
            for i in range(3):
                obstacle = rep.create.cube(
                    semantics=[("class", "obstacle")],
                    scale=rep.distribution.uniform(0.3, 0.8)
                )
                with rep.trigger.on_frame():
                    with obstacle:
                        rep.modify.pose(
                            position=rep.distribution.uniform(
                                (-2, -2, 0),
                                (2, 2, 0)
                            )
                        )

    def generate_all(self, frames_per_scenario: int = 500):
        """Generate all scenarios."""
        for scenario in self.scenarios:
            print(f"Generating {scenario} scenario...")

            if scenario == "handover":
                self.create_handover_scenario()
            elif scenario == "navigation":
                self.create_navigation_scenario()
            # Add other scenarios...

            # Camera
            camera = rep.create.camera()
            with rep.trigger.on_frame(num_frames=frames_per_scenario):
                with camera:
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (2, -2, 1.5),
                            (4, 2, 3)
                        ),
                        look_at=(0, 0, 1)
                    )

            # Setup writer
            render_product = rep.create.render_product(camera, (640, 480))
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(
                output_dir=f"{self.output_dir}/{scenario}",
                rgb=True,
                bounding_box_2d_tight=True,
                semantic_segmentation=True
            )
            writer.attach([render_product])

            rep.orchestrator.run()

# Generate scenarios
generator = ScenarioGenerator("_output/scenarios")
generator.generate_all()

simulation_app.close()
```

---

## 10.6 Data Quality Validation

### Dataset Analyzer

```python
# dataset_analyzer.py
import json
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

class DatasetAnalyzer:
    """Analyze synthetic dataset quality."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.annotations = self.load_annotations()

    def load_annotations(self):
        """Load COCO annotations."""
        ann_file = os.path.join(self.dataset_path, "annotations.json")
        with open(ann_file, 'r') as f:
            return json.load(f)

    def class_distribution(self):
        """Analyze class distribution."""
        classes = []
        for ann in self.annotations.get("annotations", []):
            classes.append(ann.get("category_id", 0))

        counter = Counter(classes)

        # Map to class names
        categories = {
            c["id"]: c["name"]
            for c in self.annotations.get("categories", [])
        }

        print("\nClass Distribution:")
        for class_id, count in counter.most_common():
            name = categories.get(class_id, f"Unknown_{class_id}")
            print(f"  {name}: {count}")

        return counter

    def bbox_size_distribution(self):
        """Analyze bounding box sizes."""
        areas = []
        aspect_ratios = []

        for ann in self.annotations.get("annotations", []):
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                w, h = bbox[2], bbox[3]
                areas.append(w * h)
                if h > 0:
                    aspect_ratios.append(w / h)

        print(f"\nBBox Statistics:")
        print(f"  Area - Mean: {np.mean(areas):.1f}, Std: {np.std(areas):.1f}")
        print(f"  Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}")

        return areas, aspect_ratios

    def occlusion_analysis(self):
        """Analyze object occlusion levels."""
        occlusion_levels = []

        for ann in self.annotations.get("annotations", []):
            # Compare tight vs loose bbox
            tight = ann.get("bbox_tight", [0, 0, 0, 0])
            loose = ann.get("bbox_loose", [0, 0, 0, 0])

            if len(tight) >= 4 and len(loose) >= 4:
                tight_area = tight[2] * tight[3]
                loose_area = loose[2] * loose[3]

                if loose_area > 0:
                    visibility = tight_area / loose_area
                    occlusion_levels.append(1 - visibility)

        if occlusion_levels:
            print(f"\nOcclusion Analysis:")
            print(f"  Mean occlusion: {np.mean(occlusion_levels):.2%}")
            print(f"  Fully visible (<10% occluded): {sum(o < 0.1 for o in occlusion_levels)}")
            print(f"  Partially occluded (10-50%): {sum(0.1 <= o < 0.5 for o in occlusion_levels)}")
            print(f"  Heavily occluded (>50%): {sum(o >= 0.5 for o in occlusion_levels)}")

        return occlusion_levels

    def generate_report(self, output_path: str = "dataset_report.html"):
        """Generate HTML quality report."""
        class_dist = self.class_distribution()
        areas, ratios = self.bbox_size_distribution()
        occlusion = self.occlusion_analysis()

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Class distribution
        axes[0, 0].bar(class_dist.keys(), class_dist.values())
        axes[0, 0].set_title("Class Distribution")

        # Area histogram
        axes[0, 1].hist(areas, bins=50)
        axes[0, 1].set_title("BBox Area Distribution")

        # Aspect ratio histogram
        axes[1, 0].hist(ratios, bins=50)
        axes[1, 0].set_title("Aspect Ratio Distribution")

        # Occlusion histogram
        if occlusion:
            axes[1, 1].hist(occlusion, bins=20)
            axes[1, 1].set_title("Occlusion Distribution")

        plt.tight_layout()
        plt.savefig(output_path.replace(".html", ".png"))
        plt.close()

        print(f"\nReport saved to {output_path}")

# Usage
analyzer = DatasetAnalyzer("_output/coco_dataset")
analyzer.generate_report()
```

---

## Exercises

### Exercise 10.1: Basic Data Generation

**Objective**: Generate RGB images with 2D bounding boxes.

**Difficulty**: Beginner | **Estimated Time**: 30 minutes

#### Instructions

1. Create scene with simple objects
2. Set up RGB and bbox annotation
3. Generate 100 frames
4. Verify output format

---

### Exercise 10.2: Domain Randomization

**Objective**: Apply lighting and texture randomization.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Randomize light position and intensity
2. Randomize object materials
3. Add distractor objects
4. Generate diverse dataset

---

### Exercise 10.3: Pose Estimation Dataset

**Objective**: Create humanoid pose estimation dataset.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Load humanoid model with skeleton
2. Configure keypoint annotations
3. Generate varied poses
4. Export in COCO keypoint format

---

### Exercise 10.4: Dataset Quality Analysis

**Objective**: Validate generated dataset quality.

**Difficulty**: Advanced | **Estimated Time**: 45 minutes

#### Instructions

1. Run dataset analyzer
2. Check class balance
3. Verify annotation accuracy
4. Generate quality report

---

## Summary

In this chapter, you learned:

- **Synthetic data** provides unlimited labeled training samples
- **Domain randomization** improves sim-to-real transfer
- **Multiple annotation types** support various perception tasks
- **Scenario generation** creates diverse training conditions
- **Quality validation** ensures dataset utility

---

## References

[1] J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," in *IROS*, 2017.

[2] NVIDIA, "Isaac Sim Replicator," [Online]. Available: https://docs.omniverse.nvidia.com/isaacsim/latest/replicator.html.

[3] T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," in *ECCV*, 2014.

[4] J. Tremblay et al., "Training Deep Networks with Synthetic Data: Bridging the Reality Gap," in *CVPR Workshops*, 2018.
