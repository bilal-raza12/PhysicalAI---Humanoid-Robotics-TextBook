---
id: ch08-unity-hri
title: "Chapter 8: Unity for High-Fidelity HRI Scenes"
sidebar_position: 5
---

# Chapter 8: Unity for High-Fidelity Human-Robot Interaction Scenes

**Estimated Time**: 5-6 hours | **Exercises**: 4

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Set up** Unity with ROS 2 integration for robotics simulation
2. **Create** photorealistic environments for HRI testing
3. **Implement** virtual human characters for interaction scenarios
4. **Design** multimodal interaction interfaces (speech, gesture, gaze)
5. **Evaluate** HRI experiments in simulation

---

## 8.1 Introduction to Unity for Robotics

Unity provides high-fidelity graphics and physics for human-robot interaction research.

### Why Unity for HRI?

| Feature | Benefit for HRI |
|---------|-----------------|
| Photorealistic rendering | Realistic human appearance |
| Animation system | Natural human motion |
| VR/AR support | Immersive telepresence |
| Cross-platform | Deploy to various devices |
| Asset Store | Pre-built characters and environments |

### Unity Robotics Hub

```bash
# Install Unity Hub
# Download from: https://unity.com/download

# Create new Unity project (2022.3 LTS recommended)
# Add packages via Package Manager:
# - ROS TCP Connector
# - URDF Importer
# - Perception (for synthetic data)
```

### ROS 2 Integration Setup

```csharp
// ROS2Connection.cs
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using UnityEngine;

public class ROS2Connection : MonoBehaviour
{
    private ROSConnection ros;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<StringMsg>("/unity/status");

        // Subscribe to robot commands
        ros.Subscribe<StringMsg>("/robot/speech", OnSpeechReceived);

        Debug.Log("ROS 2 connection established");
    }

    void OnSpeechReceived(StringMsg msg)
    {
        Debug.Log($"Robot says: {msg.data}");
        // Trigger text-to-speech or display
    }

    public void PublishStatus(string status)
    {
        var msg = new StringMsg { data = status };
        ros.Publish("/unity/status", msg);
    }
}
```

---

## 8.2 Importing Robot Models

### URDF Import Process

```csharp
// RobotImporter.cs
using Unity.Robotics.UrdfImporter;
using UnityEngine;

public class RobotImporter : MonoBehaviour
{
    [SerializeField] private string urdfPath = "Assets/Robots/humanoid.urdf";

    void Start()
    {
        ImportRobot();
    }

    void ImportRobot()
    {
        // Import settings
        var settings = new ImportSettings
        {
            convexMethod = ImportSettings.ConvexDecomposer.VHACD,
            chosenAxis = ImportSettings.axisType.yAxis
        };

        // Import URDF
        UrdfRobotExtensions.Create(urdfPath, settings);

        Debug.Log("Robot imported successfully");
    }
}
```

### Joint Control Interface

```csharp
// JointController.cs
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using UnityEngine;
using System.Collections.Generic;

public class JointController : MonoBehaviour
{
    private ROSConnection ros;
    private Dictionary<string, ArticulationBody> joints;

    [SerializeField] private string jointStateTopic = "/joint_states";

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<JointStateMsg>(jointStateTopic, OnJointStateReceived);

        // Find all articulation bodies
        joints = new Dictionary<string, ArticulationBody>();
        foreach (var joint in GetComponentsInChildren<ArticulationBody>())
        {
            joints[joint.name] = joint;
        }
    }

    void OnJointStateReceived(JointStateMsg msg)
    {
        for (int i = 0; i < msg.name.Length; i++)
        {
            string jointName = msg.name[i];
            if (joints.ContainsKey(jointName))
            {
                SetJointTarget(joints[jointName], (float)msg.position[i]);
            }
        }
    }

    void SetJointTarget(ArticulationBody joint, float target)
    {
        var drive = joint.xDrive;
        drive.target = target * Mathf.Rad2Deg;
        joint.xDrive = drive;
    }
}
```

---

## 8.3 Creating Virtual Humans

### Character Setup with Animation

```csharp
// VirtualHuman.cs
using UnityEngine;

public class VirtualHuman : MonoBehaviour
{
    private Animator animator;

    [Header("Animation Parameters")]
    public float walkSpeed = 1.0f;
    public float emotionIntensity = 0.5f;

    [Header("Interaction")]
    public Transform gazeTarget;
    public bool isInteracting = false;

    void Start()
    {
        animator = GetComponent<Animator>();
    }

    void Update()
    {
        UpdateGaze();
        UpdateLocomotion();
    }

    void UpdateGaze()
    {
        if (gazeTarget != null)
        {
            // Look at target
            Vector3 lookDir = gazeTarget.position - transform.position;
            lookDir.y = 0;

            if (lookDir.magnitude > 0.1f)
            {
                Quaternion targetRot = Quaternion.LookRotation(lookDir);
                transform.rotation = Quaternion.Slerp(
                    transform.rotation,
                    targetRot,
                    Time.deltaTime * 2.0f
                );
            }
        }
    }

    void UpdateLocomotion()
    {
        animator.SetFloat("Speed", walkSpeed);
        animator.SetBool("IsInteracting", isInteracting);
    }

    public void SetEmotion(string emotion, float intensity)
    {
        // Blend shapes for facial expressions
        animator.SetFloat("EmotionIntensity", intensity);

        switch (emotion.ToLower())
        {
            case "happy":
                animator.SetTrigger("Happy");
                break;
            case "sad":
                animator.SetTrigger("Sad");
                break;
            case "surprised":
                animator.SetTrigger("Surprised");
                break;
            case "neutral":
                animator.SetTrigger("Neutral");
                break;
        }
    }

    public void Wave()
    {
        animator.SetTrigger("Wave");
    }

    public void Point(Vector3 target)
    {
        animator.SetTrigger("Point");
        // IK for pointing direction
    }
}
```

### Crowd Simulation

```csharp
// CrowdManager.cs
using UnityEngine;
using UnityEngine.AI;
using System.Collections.Generic;

public class CrowdManager : MonoBehaviour
{
    [SerializeField] private GameObject humanPrefab;
    [SerializeField] private int crowdSize = 20;
    [SerializeField] private float spawnRadius = 10f;

    private List<NavMeshAgent> agents = new List<NavMeshAgent>();

    void Start()
    {
        SpawnCrowd();
    }

    void SpawnCrowd()
    {
        for (int i = 0; i < crowdSize; i++)
        {
            Vector3 randomPos = Random.insideUnitSphere * spawnRadius;
            randomPos.y = 0;

            NavMeshHit hit;
            if (NavMesh.SamplePosition(randomPos, out hit, spawnRadius, NavMesh.AllAreas))
            {
                GameObject human = Instantiate(humanPrefab, hit.position, Quaternion.identity);
                NavMeshAgent agent = human.GetComponent<NavMeshAgent>();
                agents.Add(agent);

                // Set random destination
                SetRandomDestination(agent);
            }
        }
    }

    void SetRandomDestination(NavMeshAgent agent)
    {
        Vector3 randomDir = Random.insideUnitSphere * spawnRadius;
        randomDir += transform.position;

        NavMeshHit hit;
        if (NavMesh.SamplePosition(randomDir, out hit, spawnRadius, NavMesh.AllAreas))
        {
            agent.SetDestination(hit.position);
        }
    }

    void Update()
    {
        foreach (var agent in agents)
        {
            if (!agent.pathPending && agent.remainingDistance < 0.5f)
            {
                SetRandomDestination(agent);
            }
        }
    }
}
```

---

## 8.4 Multimodal Interaction

### Speech Integration

```csharp
// SpeechInterface.cs
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using UnityEngine;

public class SpeechInterface : MonoBehaviour
{
    private ROSConnection ros;

    [Header("Text-to-Speech")]
    public AudioSource audioSource;

    [Header("Speech Recognition")]
    public bool isListening = false;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Subscribe to robot speech output
        ros.Subscribe<StringMsg>("/robot/tts", OnTTSReceived);

        // Publisher for recognized speech
        ros.RegisterPublisher<StringMsg>("/human/speech");
    }

    void OnTTSReceived(StringMsg msg)
    {
        // In production, use a TTS service
        Debug.Log($"Robot TTS: {msg.data}");
        DisplaySubtitle(msg.data);
    }

    void DisplaySubtitle(string text)
    {
        // Display subtitle UI
        // UIManager.Instance.ShowSubtitle(text);
    }

    public void OnSpeechRecognized(string text)
    {
        var msg = new StringMsg { data = text };
        ros.Publish("/human/speech", msg);

        Debug.Log($"Human said: {text}");
    }

    // Called by microphone input system
    public void StartListening()
    {
        isListening = true;
        // Start speech recognition
    }

    public void StopListening()
    {
        isListening = false;
        // Process audio and get transcription
    }
}
```

### Gesture Recognition

```csharp
// GestureRecognizer.cs
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using UnityEngine;

public class GestureRecognizer : MonoBehaviour
{
    private ROSConnection ros;

    [Header("Hand Tracking")]
    public Transform leftHand;
    public Transform rightHand;
    public Transform head;

    private Vector3 lastLeftPos;
    private Vector3 lastRightPos;
    private float gestureThreshold = 0.5f;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<StringMsg>("/human/gesture");

        lastLeftPos = leftHand.position;
        lastRightPos = rightHand.position;
    }

    void Update()
    {
        DetectGestures();
    }

    void DetectGestures()
    {
        // Wave detection
        Vector3 leftVelocity = (leftHand.position - lastLeftPos) / Time.deltaTime;
        Vector3 rightVelocity = (rightHand.position - lastRightPos) / Time.deltaTime;

        // Check for wave gesture (hand above head, moving side to side)
        if (rightHand.position.y > head.position.y &&
            Mathf.Abs(rightVelocity.x) > gestureThreshold)
        {
            PublishGesture("wave");
        }

        // Check for pointing gesture
        if (IsPointing(rightHand))
        {
            Vector3 pointDir = rightHand.forward;
            PublishGesture($"point:{pointDir.x},{pointDir.y},{pointDir.z}");
        }

        // Check for stop gesture (palm facing forward)
        if (IsPalmForward(rightHand) && rightHand.position.y > head.position.y * 0.8f)
        {
            PublishGesture("stop");
        }

        lastLeftPos = leftHand.position;
        lastRightPos = rightHand.position;
    }

    bool IsPointing(Transform hand)
    {
        // Simplified pointing detection
        // In production, use hand tracking skeleton
        return false;
    }

    bool IsPalmForward(Transform hand)
    {
        return Vector3.Dot(hand.forward, Camera.main.transform.forward) < -0.7f;
    }

    void PublishGesture(string gesture)
    {
        var msg = new StringMsg { data = gesture };
        ros.Publish("/human/gesture", msg);
        Debug.Log($"Gesture detected: {gesture}");
    }
}
```

### Gaze Tracking

```csharp
// GazeTracker.cs
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using UnityEngine;

public class GazeTracker : MonoBehaviour
{
    private ROSConnection ros;

    [Header("Gaze Settings")]
    public Transform headTransform;
    public LayerMask gazeLayers;
    public float maxGazeDistance = 10f;

    [Header("Debug")]
    public bool showGazeRay = true;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<PointStampedMsg>("/human/gaze_point");
    }

    void Update()
    {
        TrackGaze();
    }

    void TrackGaze()
    {
        Ray gazeRay = new Ray(headTransform.position, headTransform.forward);
        RaycastHit hit;

        if (Physics.Raycast(gazeRay, out hit, maxGazeDistance, gazeLayers))
        {
            PublishGazePoint(hit.point);

            if (showGazeRay)
            {
                Debug.DrawLine(headTransform.position, hit.point, Color.green);
            }

            // Check if looking at robot
            if (hit.collider.CompareTag("Robot"))
            {
                OnGazeAtRobot(hit.collider.gameObject);
            }
        }
        else if (showGazeRay)
        {
            Debug.DrawRay(headTransform.position, headTransform.forward * maxGazeDistance, Color.red);
        }
    }

    void PublishGazePoint(Vector3 point)
    {
        var msg = new PointStampedMsg
        {
            header = new RosMessageTypes.Std.HeaderMsg
            {
                frame_id = "world"
            },
            point = new PointMsg
            {
                x = point.x,
                y = point.z,  // Unity Y -> ROS Z
                z = point.y   // Unity Z -> ROS Y
            }
        };

        ros.Publish("/human/gaze_point", msg);
    }

    void OnGazeAtRobot(GameObject robot)
    {
        // Trigger robot attention
        Debug.Log("Human is looking at robot");
    }
}
```

---

## 8.5 HRI Experiment Framework

### Experiment Controller

```csharp
// ExperimentController.cs
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;

public class ExperimentController : MonoBehaviour
{
    private ROSConnection ros;

    [Header("Experiment Settings")]
    public string experimentId;
    public string participantId;
    public int currentTrial = 0;
    public int totalTrials = 10;

    [Header("Data Logging")]
    public bool logData = true;
    public string logDirectory = "ExperimentData";

    private List<TrialData> trialData = new List<TrialData>();
    private DateTime experimentStartTime;
    private DateTime trialStartTime;

    [Serializable]
    public class TrialData
    {
        public int trialNumber;
        public float duration;
        public string condition;
        public float taskCompletionTime;
        public int errorCount;
        public List<string> events;
    }

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<StringMsg>("/experiment/status");
        ros.Subscribe<StringMsg>("/experiment/command", OnExperimentCommand);

        experimentStartTime = DateTime.Now;
        experimentId = $"EXP_{experimentStartTime:yyyyMMdd_HHmmss}";

        Debug.Log($"Experiment {experimentId} initialized");
    }

    void OnExperimentCommand(StringMsg msg)
    {
        string[] parts = msg.data.Split(':');
        string command = parts[0];

        switch (command)
        {
            case "start":
                StartExperiment();
                break;
            case "next_trial":
                NextTrial();
                break;
            case "end":
                EndExperiment();
                break;
            case "log_event":
                if (parts.Length > 1) LogEvent(parts[1]);
                break;
        }
    }

    public void StartExperiment()
    {
        currentTrial = 0;
        trialData.Clear();
        PublishStatus("experiment_started");
        StartTrial();
    }

    public void StartTrial()
    {
        trialStartTime = DateTime.Now;
        var trial = new TrialData
        {
            trialNumber = currentTrial,
            events = new List<string>()
        };
        trialData.Add(trial);

        PublishStatus($"trial_started:{currentTrial}");
        Debug.Log($"Trial {currentTrial} started");
    }

    public void EndTrial()
    {
        var trial = trialData[currentTrial];
        trial.duration = (float)(DateTime.Now - trialStartTime).TotalSeconds;

        PublishStatus($"trial_ended:{currentTrial}");
        Debug.Log($"Trial {currentTrial} ended. Duration: {trial.duration}s");
    }

    public void NextTrial()
    {
        EndTrial();
        currentTrial++;

        if (currentTrial < totalTrials)
        {
            StartTrial();
        }
        else
        {
            EndExperiment();
        }
    }

    public void EndExperiment()
    {
        PublishStatus("experiment_ended");
        SaveData();
        Debug.Log("Experiment completed");
    }

    public void LogEvent(string eventName)
    {
        if (currentTrial < trialData.Count)
        {
            string timestamp = (DateTime.Now - trialStartTime).TotalSeconds.ToString("F3");
            trialData[currentTrial].events.Add($"{timestamp}:{eventName}");
        }
    }

    void SaveData()
    {
        if (!logData) return;

        string path = Path.Combine(logDirectory, $"{experimentId}_{participantId}.json");
        Directory.CreateDirectory(logDirectory);

        string json = JsonUtility.ToJson(new ExperimentData
        {
            experimentId = experimentId,
            participantId = participantId,
            trials = trialData
        }, true);

        File.WriteAllText(path, json);
        Debug.Log($"Data saved to {path}");
    }

    void PublishStatus(string status)
    {
        var msg = new StringMsg { data = status };
        ros.Publish("/experiment/status", msg);
    }

    [Serializable]
    private class ExperimentData
    {
        public string experimentId;
        public string participantId;
        public List<TrialData> trials;
    }
}
```

### Questionnaire System

```csharp
// QuestionnaireUI.cs
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class QuestionnaireUI : MonoBehaviour
{
    [Header("UI Elements")]
    public GameObject questionPanel;
    public Text questionText;
    public Slider likertSlider;
    public Button nextButton;
    public Text scaleMinLabel;
    public Text scaleMaxLabel;

    [Header("Questions")]
    public List<Question> questions;
    private int currentQuestion = 0;
    private List<int> responses = new List<int>();

    [System.Serializable]
    public class Question
    {
        public string text;
        public string minLabel = "Strongly Disagree";
        public string maxLabel = "Strongly Agree";
        public int minValue = 1;
        public int maxValue = 7;
    }

    void Start()
    {
        nextButton.onClick.AddListener(OnNextClicked);
        Hide();
    }

    public void Show()
    {
        questionPanel.SetActive(true);
        currentQuestion = 0;
        responses.Clear();
        DisplayQuestion();
    }

    public void Hide()
    {
        questionPanel.SetActive(false);
    }

    void DisplayQuestion()
    {
        if (currentQuestion >= questions.Count)
        {
            OnQuestionnaireComplete();
            return;
        }

        var q = questions[currentQuestion];
        questionText.text = q.text;
        scaleMinLabel.text = q.minLabel;
        scaleMaxLabel.text = q.maxLabel;
        likertSlider.minValue = q.minValue;
        likertSlider.maxValue = q.maxValue;
        likertSlider.value = (q.minValue + q.maxValue) / 2f;
    }

    void OnNextClicked()
    {
        responses.Add((int)likertSlider.value);
        currentQuestion++;
        DisplayQuestion();
    }

    void OnQuestionnaireComplete()
    {
        Hide();

        // Calculate scores
        float average = 0;
        foreach (int r in responses) average += r;
        average /= responses.Count;

        Debug.Log($"Questionnaire complete. Average: {average}");

        // Send to experiment controller
        var controller = FindObjectOfType<ExperimentController>();
        if (controller != null)
        {
            controller.LogEvent($"questionnaire_complete:{average}");
        }
    }
}
```

---

## 8.6 Environment Design

### Indoor Scene Setup

```csharp
// EnvironmentManager.cs
using UnityEngine;
using System.Collections.Generic;

public class EnvironmentManager : MonoBehaviour
{
    [Header("Scene Elements")]
    public List<GameObject> furniturePrefabs;
    public List<GameObject> propPrefabs;
    public Transform spawnArea;

    [Header("Lighting")]
    public Light mainLight;
    public float dayNightCycle = 0f; // 0-1

    [Header("Audio")]
    public AudioSource ambientAudio;
    public List<AudioClip> ambientSounds;

    void Start()
    {
        SetupEnvironment();
    }

    void SetupEnvironment()
    {
        // Randomize lighting for varied training data
        UpdateLighting();

        // Play ambient sounds
        if (ambientSounds.Count > 0)
        {
            ambientAudio.clip = ambientSounds[Random.Range(0, ambientSounds.Count)];
            ambientAudio.Play();
        }
    }

    void UpdateLighting()
    {
        // Simulate time of day
        float angle = dayNightCycle * 180f;
        mainLight.transform.rotation = Quaternion.Euler(angle, -30f, 0f);

        // Adjust intensity
        mainLight.intensity = Mathf.Sin(dayNightCycle * Mathf.PI) * 0.8f + 0.2f;

        // Adjust color temperature
        float temp = Mathf.Lerp(4000f, 6500f, dayNightCycle);
        mainLight.colorTemperature = temp;
    }

    public void RandomizeEnvironment()
    {
        // Move furniture slightly
        foreach (var furniture in GameObject.FindGameObjectsWithTag("Furniture"))
        {
            Vector3 offset = Random.insideUnitSphere * 0.1f;
            offset.y = 0;
            furniture.transform.position += offset;

            float rotOffset = Random.Range(-5f, 5f);
            furniture.transform.Rotate(0, rotOffset, 0);
        }

        // Change lighting
        dayNightCycle = Random.Range(0.2f, 0.8f);
        UpdateLighting();
    }
}
```

---

## Exercises

### Exercise 8.1: Unity-ROS 2 Setup

**Objective**: Establish communication between Unity and ROS 2.

**Difficulty**: Beginner | **Estimated Time**: 45 minutes

#### Instructions

1. Install Unity 2022.3 LTS
2. Add ROS TCP Connector package
3. Configure ROS endpoint
4. Test bidirectional messaging

#### Expected Outcome

Unity sending/receiving ROS 2 messages successfully.

---

### Exercise 8.2: Import Humanoid Robot

**Objective**: Import and control a humanoid URDF in Unity.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Export humanoid URDF from ROS 2 workspace
2. Import using URDF Importer
3. Set up articulation bodies
4. Control joints via ROS 2 joint_states

---

### Exercise 8.3: Virtual Human Interaction

**Objective**: Create an interactive virtual human character.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Import humanoid character from Asset Store
2. Set up animator controller
3. Implement gaze following
4. Add gesture animations

---

### Exercise 8.4: HRI Experiment

**Objective**: Design and run a simple HRI experiment.

**Difficulty**: Advanced | **Estimated Time**: 90 minutes

#### Instructions

1. Create experiment scenario
2. Implement data logging
3. Add post-trial questionnaire
4. Analyze collected data

---

## Summary

In this chapter, you learned:

- **Unity integration** with ROS 2 enables high-fidelity HRI simulation
- **Virtual humans** can be animated and controlled for interaction studies
- **Multimodal interfaces** capture speech, gesture, and gaze
- **Experiment frameworks** enable systematic HRI research
- **Environment design** affects the realism of simulations

---

## References

[1] Unity Technologies, "Unity Robotics Hub," [Online]. Available: https://github.com/Unity-Technologies/Unity-Robotics-Hub.

[2] M. L. Walters et al., "A framework for robot human interaction experiments," in *IEEE Int. Conf. Robot. Autom.*, 2005.

[3] C. Bartneck et al., "Measurement Instruments for the Anthropomorphism, Animacy, Likeability, Perceived Intelligence, and Perceived Safety of Robots," *Int. J. Soc. Robot.*, 2009.

[4] Unity Technologies, "Perception Package," [Online]. Available: https://github.com/Unity-Technologies/com.unity.perception.
