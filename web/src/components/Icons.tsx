// Small line icons, drawn in the current text colour.

const base = {
  width: 22,
  height: 22,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export const WordsIcon = () => (
  <svg {...base}>
    <path d="M4 5h16v11H9l-5 4z" />
    <path d="M8 9.5h8M8 12.5h5" />
  </svg>
);

export const VoiceIcon = () => (
  <svg {...base}>
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
  </svg>
);

export const FaceIcon = () => (
  <svg {...base}>
    <circle cx="12" cy="12" r="9" />
    <path d="M9 10h.01M15 10h.01M8.5 14.5c1.8 1.8 5.2 1.8 7 0" />
  </svg>
);

export const PulseIcon = () => (
  <svg {...base}>
    <path d="M3 12h4l2-5 4 10 2-5h6" />
  </svg>
);

export const ClockIcon = () => (
  <svg {...base}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </svg>
);

export const SkipIcon = () => (
  <svg {...base}>
    <path d="M5 12h12M13 8l4 4-4 4M19 6v12" />
  </svg>
);

export const LockIcon = () => (
  <svg {...base}>
    <rect x="5" y="11" width="14" height="9" rx="2" />
    <path d="M8 11V8a4 4 0 0 1 8 0v3" />
  </svg>
);

export const CheckIcon = () => (
  <svg {...base} width={18} height={18}>
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </svg>
);

export const Logo = ({ size = 30 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
    <rect width="32" height="32" rx="9" fill="var(--brand)" />
    <path d="M7 18c3-5 6-5 9 0s6 5 9 0" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" />
    <path d="M7 12.5c3-4 6-4 9 0s6 4 9 0" fill="none" stroke="#fff" strokeOpacity=".55" strokeWidth="2" strokeLinecap="round" />
  </svg>
);
