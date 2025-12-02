/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark theme (trading terminal style)
        dark: {
          bg: {
            primary: '#0a0e14',
            secondary: '#0f1419',
            card: '#151b23',
            elevated: '#1c2430',
          },
          border: '#2d3748',
          text: {
            primary: '#e6edf3',
            secondary: '#8b949e',
            muted: '#6b7280',
          },
        },
        // Light theme
        light: {
          bg: {
            primary: '#f8fafc',
            secondary: '#f1f5f9',
            card: '#ffffff',
            elevated: '#ffffff',
          },
          border: '#e2e8f0',
          text: {
            primary: '#0f172a',
            secondary: '#475569',
            muted: '#94a3b8',
          },
        },
        // Trading colors
        profit: {
          DEFAULT: '#10b981',
          light: '#34d399',
          dark: '#059669',
        },
        loss: {
          DEFAULT: '#ef4444',
          light: '#f87171',
          dark: '#dc2626',
        },
        // Accent colors
        accent: {
          blue: '#3b82f6',
          purple: '#8b5cf6',
          cyan: '#06b6d4',
          amber: '#f59e0b',
        },
      },
      fontFamily: {
        sans: ['JetBrains Mono', 'Fira Code', 'monospace'],
        display: ['Space Grotesk', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}

