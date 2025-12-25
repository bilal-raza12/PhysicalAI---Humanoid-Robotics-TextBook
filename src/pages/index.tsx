import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

// Module data for quick access cards
const MODULES = [
  {
    id: 1,
    title: 'ROS 2 Fundamentals',
    description: 'Master the Robot Operating System 2 - nodes, topics, services, and actions',
    icon: '🤖',
    color: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
    link: '/docs/module-1-ros2',
    chapters: 5,
  },
  {
    id: 2,
    title: 'Digital Twin & Simulation',
    description: 'Build virtual replicas in Gazebo, Unity, and create accurate simulations',
    icon: '🌐',
    color: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)',
    link: '/docs/module-2-digital-twin',
    chapters: 4,
  },
  {
    id: 3,
    title: 'NVIDIA Isaac Platform',
    description: 'Leverage GPU-accelerated simulation and AI with Isaac Sim & SDK',
    icon: '⚡',
    color: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
    link: '/docs/module-3-nvidia-isaac',
    chapters: 5,
  },
  {
    id: 4,
    title: 'Vision-Language-Action',
    description: 'Integrate VLA models for robots that understand natural language',
    icon: '🧠',
    color: 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)',
    link: '/docs/module-4-vla',
    chapters: 4,
  },
];

// Features for the feature section
const FEATURES = [
  {
    title: 'Simulation-First Approach',
    icon: '🎮',
    description: 'Build, test, and iterate in virtual environments before deploying to physical hardware. Save time and reduce costs.',
  },
  {
    title: 'Industry-Standard Tools',
    icon: '🛠️',
    description: 'Learn ROS 2, Gazebo, NVIDIA Isaac, and other tools used by leading robotics companies worldwide.',
  },
  {
    title: 'Hands-On Projects',
    icon: '🚀',
    description: 'Build a complete humanoid robot from scratch with step-by-step exercises and a capstone project.',
  },
  {
    title: 'AI-Powered Robotics',
    icon: '🤖',
    description: 'Integrate cutting-edge vision-language-action models for intelligent robot behavior.',
  },
];

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={styles.heroSection}>
      {/* Animated background elements */}
      <div className={styles.heroBackground}>
        <div className={styles.gradientOrb1}></div>
        <div className={styles.gradientOrb2}></div>
        <div className={styles.gradientOrb3}></div>
        <div className={styles.gridOverlay}></div>
      </div>

      <div className={styles.heroContent}>
        {/* Badge */}
        <div className={styles.badge}>
          <span className={styles.badgeIcon}>📚</span>
          <span>Open Source Textbook</span>
        </div>

        {/* Main Title */}
        <h1 className={styles.heroTitle}>
          <span className={styles.titleLine1}>Physical AI &</span>
          <span className={styles.titleLine2}>Humanoid Robotics</span>
        </h1>

        {/* Subtitle */}
        <p className={styles.heroSubtitle}>
          {siteConfig.tagline}
        </p>

        {/* Stats */}
        <div className={styles.stats}>
          <div className={styles.stat}>
            <span className={styles.statNumber}>4</span>
            <span className={styles.statLabel}>Modules</span>
          </div>
          <div className={styles.statDivider}></div>
          <div className={styles.stat}>
            <span className={styles.statNumber}>18+</span>
            <span className={styles.statLabel}>Chapters</span>
          </div>
          <div className={styles.statDivider}></div>
          <div className={styles.stat}>
            <span className={styles.statNumber}>50+</span>
            <span className={styles.statLabel}>Exercises</span>
          </div>
        </div>

        {/* CTA Buttons */}
        <div className={styles.ctaButtons}>
          <Link className={styles.primaryButton} to="docs">
            <span>Start Learning</span>
            <span className={styles.buttonIcon}>→</span>
          </Link>
          <Link className={styles.secondaryButton} to="docs/prerequisites">
            <span className={styles.buttonIcon}>📋</span>
            <span>Prerequisites</span>
          </Link>
        </div>

        {/* Hero Image/Robot Illustration */}
        <div className={styles.heroVisual}>
          {/* Tech Stack Labels */}
          <div className={styles.techLabels}>
            <div className={`${styles.techLabel} ${styles.techLabelLeft1}`}>
              <span className={styles.techDot}></span>
              <span className={styles.techLine}></span>
              <div className={styles.techContent}>
                <span className={styles.techIcon}>🤖</span>
                <span className={styles.techText}>ROS 2 Humble</span>
              </div>
            </div>
            <div className={`${styles.techLabel} ${styles.techLabelLeft2}`}>
              <span className={styles.techDot}></span>
              <span className={styles.techLine}></span>
              <div className={styles.techContent}>
                <span className={styles.techIcon}>🎮</span>
                <span className={styles.techText}>Isaac Sim</span>
              </div>
            </div>
            <div className={`${styles.techLabel} ${styles.techLabelRight1}`}>
              <div className={styles.techContent}>
                <span className={styles.techText}>VLA Model</span>
                <span className={styles.techIcon}>🧠</span>
              </div>
              <span className={styles.techLineRight}></span>
              <span className={styles.techDot}></span>
            </div>
            <div className={`${styles.techLabel} ${styles.techLabelRight2}`}>
              <div className={styles.techContent}>
                <span className={styles.techText}>Digital Twin</span>
                <span className={styles.techIcon}>🌐</span>
              </div>
              <span className={styles.techLineRight}></span>
              <span className={styles.techDot}></span>
            </div>
          </div>

          {/* Main Robot */}
          <div className={styles.robotContainer}>
            <div className={styles.robotGlow}></div>
            <div className={styles.robotPlatform}></div>

            {/* Humanoid Robot */}
            <div className={styles.humanoidRobot}>
              {/* Head */}
              <div className={styles.robotHead}>
                <div className={styles.headVisor}>
                  <div className={styles.visorGlow}></div>
                </div>
                <div className={styles.headAntenna}></div>
              </div>

              {/* Neck */}
              <div className={styles.robotNeck}></div>

              {/* Torso */}
              <div className={styles.robotTorso}>
                <div className={styles.torsoPlate}></div>
                <div className={styles.coreReactor}>
                  <div className={styles.coreInner}></div>
                  <div className={styles.coreRing}></div>
                </div>
                <div className={styles.torsoVents}>
                  <span></span><span></span><span></span>
                </div>
              </div>

              {/* Arms */}
              <div className={styles.robotArms}>
                <div className={`${styles.arm} ${styles.armLeft}`}>
                  <div className={styles.shoulder}></div>
                  <div className={styles.upperArm}></div>
                  <div className={styles.elbow}></div>
                  <div className={styles.forearm}></div>
                  <div className={styles.hand}>✋</div>
                </div>
                <div className={`${styles.arm} ${styles.armRight}`}>
                  <div className={styles.shoulder}></div>
                  <div className={styles.upperArm}></div>
                  <div className={styles.elbow}></div>
                  <div className={styles.forearm}></div>
                  <div className={styles.hand}>🤚</div>
                </div>
              </div>
            </div>

            {/* Particle Effects */}
            <div className={styles.particles}>
              <span></span><span></span><span></span>
              <span></span><span></span><span></span>
            </div>
          </div>

          {/* Floating Code Snippets */}
          <div className={styles.codeSnippets}>
            <div className={styles.codeSnippet}>
              <code>ros2 run humanoid_robot main</code>
            </div>
            <div className={styles.codeSnippet2}>
              <code>model.predict(vision_input)</code>
            </div>
          </div>

          {/* Orbit Ring */}
          <div className={styles.orbitRing}>
            <div className={styles.orbitDot}></div>
          </div>
        </div>
      </div>

      {/* Scroll indicator */}
      <div className={styles.scrollIndicator}>
        <span>Explore Modules</span>
        <div className={styles.scrollArrow}>↓</div>
      </div>
    </header>
  );
}

function ModuleCard({ module }) {
  return (
    <Link to={module.link} className={styles.moduleCard}>
      <div className={styles.moduleIcon} style={{ background: module.color }}>
        <span>{module.icon}</span>
      </div>
      <div className={styles.moduleContent}>
        <div className={styles.moduleHeader}>
          <span className={styles.moduleNumber}>Module {module.id}</span>
          <span className={styles.moduleChapters}>{module.chapters} Chapters</span>
        </div>
        <h3 className={styles.moduleTitle}>{module.title}</h3>
        <p className={styles.moduleDescription}>{module.description}</p>
        <div className={styles.moduleAction}>
          <span>Start Learning</span>
          <span>→</span>
        </div>
      </div>
    </Link>
  );
}

function ModulesSection() {
  return (
    <section className={styles.modulesSection}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <span className={styles.sectionBadge}>📖 Curriculum</span>
          <h2 className={styles.sectionTitle}>Course Modules</h2>
          <p className={styles.sectionSubtitle}>
            Master humanoid robotics through our comprehensive, hands-on curriculum
          </p>
        </div>
        <div className={styles.modulesGrid}>
          {MODULES.map((module) => (
            <ModuleCard key={module.id} module={module} />
          ))}
        </div>
      </div>
    </section>
  );
}

function FeatureCard({ feature }) {
  return (
    <div className={styles.featureCard}>
      <div className={styles.featureIcon}>{feature.icon}</div>
      <h3 className={styles.featureTitle}>{feature.title}</h3>
      <p className={styles.featureDescription}>{feature.description}</p>
    </div>
  );
}

function FeaturesSection() {
  return (
    <section className={styles.featuresSection}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <span className={styles.sectionBadge}>✨ Why This Book</span>
          <h2 className={styles.sectionTitle}>Learn By Building</h2>
          <p className={styles.sectionSubtitle}>
            A practical, project-based approach to mastering humanoid robotics
          </p>
        </div>
        <div className={styles.featuresGrid}>
          {FEATURES.map((feature, idx) => (
            <FeatureCard key={idx} feature={feature} />
          ))}
        </div>
      </div>
    </section>
  );
}

function CapstoneSection() {
  return (
    <section className={styles.capstoneSection}>
      <div className="container">
        <div className={styles.capstoneContent}>
          <div className={styles.capstoneText}>
            <span className={styles.sectionBadge}>🎯 Capstone Project</span>
            <h2 className={styles.capstoneTitle}>Build Your Own Humanoid Robot</h2>
            <p className={styles.capstoneDescription}>
              Apply everything you've learned to build a complete, functional humanoid robot.
              From ROS 2 nodes to VLA integration - create an intelligent robot that can
              see, understand, and act in the real world.
            </p>
            <Link className={styles.capstoneButton} to="docs/capstone">
              <span>View Capstone Project</span>
              <span>→</span>
            </Link>
          </div>
          <div className={styles.capstoneVisual}>
            <div className={styles.capstoneRobot}>🦾</div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="A comprehensive textbook on Physical AI, Embodied Intelligence, and Humanoid Robotics">
      <HomepageHeader />
      <main>
        <ModulesSection />
        <FeaturesSection />
        <CapstoneSection />
      </main>
    </Layout>
  );
}
