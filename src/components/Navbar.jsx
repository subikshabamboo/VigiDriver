// components/Navbar.jsx
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ShieldCheck, Home, Activity, BarChart3, Settings, FlaskConical } from 'lucide-react';

const C = {
  navy: '#163A5F',
  amber: '#C57C21',
  border: '#E2E6EC',
  ink: '#132133',
  inkMuted: '#5B6572',
};

const LINKS = [
  { to: '/dashboard', label: 'Live Dashboard', icon: <Activity size={15} /> },
  { to: '/analytics', label: 'Analytics', icon: <BarChart3 size={15} /> },
  { to: '/settings', label: 'Settings', icon: <Settings size={15} /> },
  { to: '/sandbox', label: 'Model Sandbox', icon: <FlaskConical size={15} /> },
];

export default function NavBar() {
  const location = useLocation();

  return (
    <nav
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 24px',
        background: '#FFFFFF',
        borderBottom: `1px solid ${C.border}`,
        position: 'sticky',
        top: 0,
        zIndex: 50,
        fontFamily: "'Inter', sans-serif",
        flexWrap: 'wrap',
        gap: 10,
      }}
    >
      <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: 9, textDecoration: 'none' }}>
        <ShieldCheck size={20} color={C.amber} />
        <span
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 17,
            fontWeight: 700,
            color: C.navy,
            letterSpacing: 0.5,
          }}
        >
          VIGIDRIVER
        </span>
      </Link>

      <div
        style={{
          display: 'flex',
          gap: 4,
          background: '#F4F6F9',
          border: `1px solid ${C.border}`,
          borderRadius: 10,
          padding: 4,
          flexWrap: 'wrap',
        }}
      >
        {LINKS.map((l) => {
          const active = location.pathname === l.to;
          return (
            <Link
              key={l.to}
              to={l.to}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 7,
                padding: '9px 16px',
                borderRadius: 8,
                fontSize: 13.5,
                fontWeight: 600,
                textDecoration: 'none',
                background: active ? '#FFFFFF' : 'transparent',
                color: active ? C.navy : C.inkMuted,
                boxShadow: active ? '0 1px 4px rgba(22,58,95,0.12)' : 'none',
                transition: 'all .15s ease',
              }}
            >
              {l.icon}
              {l.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}