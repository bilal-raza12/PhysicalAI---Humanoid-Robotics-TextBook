import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A Simulation-First Approach to Building Intelligent Humanoid Systems',
  favicon: 'img/logo.svg',

  // Set the production url of your site here
  url: 'https://your-org.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: '/Physical_ai_&_Humanoid_Robotics/',

  // GitHub pages deployment config
  organizationName: 'your-org',
  projectName: 'Physical_ai_&_Humanoid_Robotics',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/your-org/Physical_ai_&_Humanoid_Robotics/tree/main/',
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.svg',
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'bookSidebar',
          position: 'left',
          label: 'Book',
        },
        {
          href: 'https://github.com/your-org/Physical_ai_&_Humanoid_Robotics',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Book',
          items: [
            {
              label: 'Introduction',
              to: 'docs',
            },
            {
              label: 'Prerequisites',
              to: 'docs/prerequisites',
            },
          ],
        },
        {
          title: 'Modules',
          items: [
            {
              label: 'Module 1: ROS 2',
              to: 'docs/module-1-ros2',
            },
            {
              label: 'Module 2: Digital Twin',
              to: 'docs/module-2-digital-twin',
            },
            {
              label: 'Module 3: NVIDIA Isaac',
              to: 'docs/module-3-nvidia-isaac',
            },
            {
              label: 'Module 4: VLA',
              to: 'docs/module-4-vla',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Capstone Project',
              to: 'docs/capstone',
            },
            {
              label: 'Appendices',
              to: 'docs/category/appendices',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/your-org/Physical_ai_Humanoid_Robotics',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'yaml', 'json', 'markup'],
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
