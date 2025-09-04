// Modern Theme System with Glassmorphism
class ThemeManager {
    constructor() {
        this.themeToggle = null;
        this.currentTheme = 'light';
        this.init();
    }

    init() {
        this.createThemeToggle();
        this.loadSavedTheme();
        this.bindEvents();
        this.applyTheme();
    }

    createThemeToggle() {
        // Create theme toggle button
        const toggle = document.createElement('button');
        toggle.className = 'theme-toggle';
        toggle.setAttribute('aria-label', 'Toggle theme');
        toggle.innerHTML = '<i class="fas fa-moon"></i>';

        // Find navbar and append to nav-content, or fallback to body
        const navContent = document.querySelector('.nav-content');
        if (navContent) {
            navContent.appendChild(toggle);
        } else {
            document.body.appendChild(toggle);
        }

        this.themeToggle = toggle;
    }

    loadSavedTheme() {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            this.currentTheme = savedTheme;
        } else {
            // Check system preference
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            this.currentTheme = prefersDark ? 'dark' : 'light';
        }
    }

    bindEvents() {
        // Theme toggle click
        this.themeToggle.addEventListener('click', () => {
            this.toggleTheme();
        });

        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!localStorage.getItem('theme')) {
                this.currentTheme = e.matches ? 'dark' : 'light';
                this.applyTheme();
            }
        });

        // Keyboard accessibility
        this.themeToggle.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.toggleTheme();
            }
        });
    }

    toggleTheme() {
        this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.applyTheme();
        this.saveTheme();
        this.animateToggle();
    }

    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.currentTheme);
        this.updateToggleIcon();
    }

    updateToggleIcon() {
        const icon = this.themeToggle.querySelector('i');
        if (this.currentTheme === 'dark') {
            icon.className = 'fas fa-sun';
        } else {
            icon.className = 'fas fa-moon';
        }
    }

    saveTheme() {
        localStorage.setItem('theme', this.currentTheme);
    }

    animateToggle() {
        // Add click animation
        this.themeToggle.style.transform = 'scale(0.95)';
        setTimeout(() => {
            this.themeToggle.style.transform = '';
        }, 150);
    }
}

// Notification System
class NotificationManager {
    static show(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas ${this.getIcon(type)}"></i>
                <span>${message}</span>
                <button class="notification-close" style="margin-left: auto; background: none; border: none; color: var(--text-primary); cursor: pointer;">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        document.body.appendChild(notification);

        // Auto remove
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, duration);

        // Close button
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            notification.remove();
        });

        return notification;
    }

    static getIcon(type) {
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        return icons[type] || icons.info;
    }
}

// Loading States
class LoadingManager {
    static show(element, text = 'Loading...') {
        const loader = document.createElement('div');
        loader.className = 'loading-overlay';
        loader.innerHTML = `
            <div style="text-align: center;">
                <div class="loading-spinner"></div>
                <p style="color: var(--text-primary); margin-top: 1rem;">${text}</p>
            </div>
        `;

        element.style.position = 'relative';
        element.appendChild(loader);
        return loader;
    }

    static hide(element) {
        const loader = element.querySelector('.loading-overlay');
        if (loader) {
            loader.remove();
        }
    }
}

// Form Enhancements
class FormEnhancer {
    static enhance() {
        // Add glass effect to form inputs
        const inputs = document.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            input.classList.add('glass-input');
        });

        // Add glass effect to buttons
        const buttons = document.querySelectorAll('button, .cta-button, .login-btn');
        buttons.forEach(button => {
            if (!button.classList.contains('theme-toggle')) {
                button.classList.add('glass-button');
            }
        });

        // Add animations to cards
        const cards = document.querySelectorAll('.feature-card, .hero, .login-container, .results-section');
        cards.forEach((card, index) => {
            card.classList.add('glass-card', 'fade-in-up');
        });
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize theme manager
    const themeManager = new ThemeManager();

    // Enhance forms
    FormEnhancer.enhance();

    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading states to forms
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function(e) {
            const submitBtn = form.querySelector('button[type="submit"], input[type="submit"]');
            if (submitBtn) {
                LoadingManager.show(form, 'Processing...');
            }
        });
    });

    // Make theme manager globally available
    window.themeManager = themeManager;
    window.NotificationManager = NotificationManager;
    window.LoadingManager = LoadingManager;
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ThemeManager, NotificationManager, LoadingManager, FormEnhancer };
}
