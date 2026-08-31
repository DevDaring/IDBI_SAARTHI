/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // SAARTHI war-room palette
        risk: {
          DEFAULT: '#F59E0B', // saffron/amber
          dark: '#D97706',
        },
        danger: {
          DEFAULT: '#DC2626',
          dark: '#991B1B',
        },
        safe: {
          DEFAULT: '#0D9488', // teal
          dark: '#0F766E',
        },
        ink: {
          DEFAULT: '#0F172A', // slate-900 navy ink
          soft: '#1E293B',
          mute: '#334155',
        },
      },
      fontFamily: {
        sans: [
          'Inter',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'sans-serif',
        ],
      },
      boxShadow: {
        card: '0 1px 3px rgba(15,23,42,0.08), 0 8px 24px rgba(15,23,42,0.06)',
        glow: '0 0 0 1px rgba(245,158,11,0.25), 0 8px 30px rgba(245,158,11,0.15)',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.55' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.4s ease-out both',
        'pulse-soft': 'pulse-soft 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
