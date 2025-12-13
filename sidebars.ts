import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Physical AI & Humanoid Robotics Textbook
 * Sidebar configuration with module-based navigation
 */
const sidebars: SidebarsConfig = {
  bookSidebar: [
    'intro',
    'prerequisites',
    'conventions',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      link: {
        type: 'doc',
        id: 'module-1-ros2/index',
      },
      items: [
        'module-1-ros2/ch01-intro-ros2',
        'module-1-ros2/ch02-nodes-topics',
        'module-1-ros2/ch03-urdf-kinematics',
        'module-1-ros2/ch04-python-agents',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      link: {
        type: 'doc',
        id: 'module-2-digital-twin/index',
      },
      items: [
        // Chapters will be added as they are created:
        // 'module-2-digital-twin/ch05-physics-sim',
        // 'module-2-digital-twin/ch06-gazebo-twin',
        // 'module-2-digital-twin/ch07-sensor-sim',
        // 'module-2-digital-twin/ch08-unity-hri',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      link: {
        type: 'doc',
        id: 'module-3-nvidia-isaac/index',
      },
      items: [
        // Chapters will be added as they are created:
        // 'module-3-nvidia-isaac/ch09-isaac-sim',
        // 'module-3-nvidia-isaac/ch10-isaac-ros',
        // 'module-3-nvidia-isaac/ch11-nav2-bipedal',
        // 'module-3-nvidia-isaac/ch12-synthetic-data',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      link: {
        type: 'doc',
        id: 'module-4-vla/index',
      },
      items: [
        // Chapters will be added as they are created:
        // 'module-4-vla/ch13-voice-action',
        // 'module-4-vla/ch14-llm-planning',
        // 'module-4-vla/ch15-perception-action',
        // 'module-4-vla/ch16-nl-execution',
      ],
    },
    {
      type: 'category',
      label: 'Capstone: The Autonomous Humanoid',
      link: {
        type: 'doc',
        id: 'capstone/index',
      },
      items: [
        // Chapters will be added as they are created:
        // 'capstone/ch17-autonomous-humanoid',
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      link: {
        type: 'generated-index',
        title: 'Appendices',
        description: 'Supplementary materials, installation guides, and reference documentation.',
      },
      items: [
        'appendices/installation',
        'appendices/hardware',
        'appendices/troubleshooting',
        'appendices/glossary',
        'appendices/bibliography',
      ],
    },
  ],
};

export default sidebars;
