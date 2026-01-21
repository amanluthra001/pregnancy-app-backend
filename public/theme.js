// Theme configuration and initialization
const lightTheme = {
  '--primary': '#6a11cb',
  '--primary2': '#2575fc',
  '--text': '#000000',
  '--muted': '#555555',
  '--bg': '#f8fafc',
  '--card': '#ffffff',
  '--border': '#e5e7eb',
};

const darkTheme = {
  '--primary': '#6a11cb',
  '--primary2': '#2575fc',
  '--text': '#f3f0ff',
  '--muted': '#bdb4e6',
  '--bg': '#18122B',
  '--card': '#23203b',
  '--border': '#393053',
};

const root = document.documentElement;
let isDark = true; // Global state

function setTheme(dark) {
  isDark = dark;
  const theme = isDark ? darkTheme : lightTheme;
  for (const key in theme) {
    root.style.setProperty(key, theme[key]);
  }
  document.body.style.background = isDark
    ? 'linear-gradient(120deg, #18122B 0%, #393053 100%)'
    : 'linear-gradient(120deg, #f8fafc 0%, #e0e7ff 100%)';
  
  // Persist theme preference
  localStorage.setItem('theme-mode', isDark ? 'dark' : 'light');
}

function initTheme() {
  // Load saved theme or default to dark
  const saved = localStorage.getItem('theme-mode');
  const dark = saved ? saved === 'dark' : true;
  setTheme(dark);
}

// Apply theme immediately (do not wait for DOMContentLoaded)
initTheme();

// Setup theme toggle when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.onclick = () => {
      setTheme(!isDark);
    };
  }
});
