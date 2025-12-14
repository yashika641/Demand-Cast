# DemandCast Design System

## 🎨 Modern UI Enhancements

### Visual Design Principles

1. **Glassmorphism** - Frosted glass effects with backdrop blur
2. **Soft Shadows** - Layered, subtle shadows for depth
3. **Gradient Accents** - Blue-to-purple gradient highlights
4. **Smooth Animations** - Spring-based transitions and hover effects
5. **Clean Typography** - Inter font family with proper hierarchy

---

## Color Palette

### Primary Colors
- **Blue 500**: `#3B82F6` - Primary actions, active states
- **Blue 600**: `#2563EB` - Hover states, emphasis

### Gradient Colors
- **Primary Gradient**: `from-blue-500 to-blue-600`
- **Background Gradient**: `from-gray-50 via-blue-50/20 to-gray-50`
- **Card Hover**: `from-blue-50/50 to-transparent`

### Semantic Colors
- **Success**: Green 50-600 range
- **Warning**: Amber 50-600 range
- **Error**: Red 50-600 range
- **Info**: Blue 50-600 range

---

## Components

### KPI Cards
- **Background**: White with soft shadow
- **Border**: `border-gray-100/50`
- **Hover Effect**: Lift animation (-2px translate-y)
- **Icon Container**: Gradient from blue-50 to blue-100
- **Trend Badges**: Bordered, backdrop-blur

### Chart Cards
- **Background**: White
- **Shadow**: Soft shadow on base, larger on hover
- **Transition**: 300ms duration
- **Header**: Semibold font, flex layout

### Modals
- **Backdrop**: `bg-black/40 backdrop-blur-sm`
- **Container**: White with shadow-2xl
- **Animation**: Spring (damping: 25, stiffness: 300)
- **Header**: Gradient from gray-50

### Sidebar
- **Background**: `bg-white/80 backdrop-blur-xl`
- **Border**: `border-gray-200/50 shadow-lg`
- **Active State**: Gradient blue background
- **Logo**: Gradient blue with shadow glow

---

## Typography

### Font Family
```css
font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
```

### Hierarchy
- **H1**: 2xl, semibold (600), -0.02em letter-spacing
- **H2**: xl, semibold (600), -0.01em letter-spacing
- **H3**: lg, semibold (600)
- **H4**: base, semibold (600)
- **Body**: base, normal (400), 1.6 line-height
- **Labels**: sm, medium (500)
- **Buttons**: sm, medium (500)

---

## Shadows

### Utilities
```css
.shadow-soft: Multi-layer soft shadow
.shadow-glow: Blue glow (20px blur, 15% opacity)
.shadow-glow-lg: Blue glow (40px blur, 20% opacity)
```

### Usage
- Cards: `shadow-soft`
- Hover states: `shadow-lg`
- Focus states: Blue glow via ring
- Modals: `shadow-2xl`

---

## Animations

### Hover Effects
```css
.hover-lift: translateY(-2px) + shadow upgrade
```

### Transitions
- **Default**: 300ms cubic-bezier(0.4, 0, 0.2, 1)
- **Spring**: damping 25, stiffness 300
- **Scale**: 110% on hover for icon buttons

### Keyframes
- **gradient**: 15s infinite background animation
- **shimmer**: 2s infinite loading effect
- **pulse-slow**: 3s infinite opacity pulse
- **badge-pulse**: 2s infinite scale pulse

---

## Spacing & Sizing

### Border Radius
- **Small**: 0.5rem (8px)
- **Medium**: 0.75rem (12px)
- **Large**: 1rem (16px)
- **XL**: 1.5rem (24px)
- **2XL**: 2rem (32px)

### Padding
- Cards: 1.5rem (24px)
- Buttons: 0.75rem 1.5rem
- Inputs: 0.75rem 1rem
- Modals: 1.5rem (24px)

---

## Interactive States

### Buttons
- **Primary**: Gradient blue, shadow on hover, -0.5px lift
- **Secondary**: White bg, gray border, gray bg on hover
- **Ghost**: Transparent, gray bg on hover
- **Ripple**: White overlay on active

### Inputs
- **Default**: White bg, gray-200 border
- **Focus**: Blue-500 border, blue-100 ring (3px)
- **Error**: Red-500 border, red-100 ring
- **Disabled**: Gray-100 bg, gray-400 text

### Links
- **Default**: Blue-600
- **Hover**: Blue-700
- **Active**: Blue-800
- **Visited**: Purple-600 (if needed)

---

## Accessibility

### Focus States
- **Outline**: 3px ring with 10% opacity color
- **Visible**: Always visible on keyboard focus
- **Color**: Matches component theme color

### Contrast Ratios
- Text on white: 4.5:1 minimum
- Interactive elements: 3:1 minimum
- Disabled states: Clearly distinguishable

---

## Responsive Design

### Breakpoints
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

### Mobile Adaptations
- Single column layouts
- Larger touch targets (44px minimum)
- Bottom navigation bar
- Hamburger menu
- Full-width cards
- Stacked KPIs

---

## Performance

### Optimizations
- Hardware-accelerated transforms
- Will-change for animations
- Backdrop-filter with fallback
- Lazy-loaded components
- Optimized font loading (preconnect)

### Best Practices
- Use transform over position changes
- Batch layout changes
- Debounce scroll/resize handlers
- Optimize images and SVGs

---

## Chart Styling

### Recharts Customization
- Grid lines: `#f1f5f9`
- Text color: `#64748b`
- Font size: 12px
- Font family: Inherited (Inter)

### Color Scheme
- Primary line: Blue-500
- Secondary line: Purple-500
- Tertiary: Green-500
- Areas: 10-30% opacity

---

## Dark Mode (Future)

### Variables Ready
- CSS custom properties defined
- Color tokens prepared
- Opacity adjustments planned

---

## Implementation Notes

### CSS Classes
All modern utilities are defined in `/styles/globals.css`:
- `.glass` - Glassmorphism effect
- `.gradient-text` - Gradient text fill
- `.animate-gradient` - Animated gradient background
- `.shimmer` - Loading shimmer
- `.hover-lift` - Lift on hover
- `.card-modern` - Modern card styling
- `.badge-modern` - Badge component

### Usage Example
```jsx
<div className="card-modern hover-lift">
  <div className="gradient-text">
    Modern UI Component
  </div>
</div>
```

---

Built with attention to detail for enterprise-grade UX
