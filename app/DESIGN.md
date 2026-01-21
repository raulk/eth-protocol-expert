# Editorial Scholar design system

A refined research interface with warm academic tones for Ethereum Protocol Intelligence.

## Aesthetic direction

**Tone:** Editorial / Academic / Scholarly
**Mood:** Warm, refined, trustworthy — like a well-designed research journal

This is NOT a typical tech dashboard. It's a scholarly tool for protocol researchers. Every element should feel considered, quiet, and professional — not flashy or startup-y.

## Typography

### Font stack
```css
--font-serif: 'Newsreader', Georgia, 'Times New Roman', serif;
--font-mono: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
```

### Usage rules
| Context | Font | Weight | Notes |
|---------|------|--------|-------|
| Body text | Serif | 400 | Primary reading font |
| Headings | Serif | 500 | Never 600+ (too heavy) |
| Code, IDs, technical | Mono | 400-500 | Accent color for links |
| Labels, badges | Mono | 400 | Uppercase + letter-spacing |

### Type scale
- Body: `1.0625rem` (17px) with `line-height: 1.75`
- H2 in prose: `1.375rem` with `-0.01em` letter-spacing
- H3 in prose: `1.125rem` with accent color
- Small/meta: `0.75rem` uppercase, `0.05em` letter-spacing

## Color palette

### Backgrounds (warm paper tones)
```css
--bg-primary: #faf8f5;    /* Main background — warm off-white */
--bg-secondary: #f5f2ed;  /* Cards, code blocks */
--bg-tertiary: #ebe7e0;   /* Hover states, borders */
--bg-elevated: #ffffff;   /* Panels, modals */
```

### Text hierarchy
```css
--text-primary: #1a1814;   /* Headings, important text */
--text-secondary: #4a4640; /* Body text */
--text-tertiary: #7a756d;  /* Secondary info */
--text-muted: #9a958d;     /* Timestamps, metadata */
```

### Accent (deep scholarly blue)
```css
--accent: #2d4a6f;
--accent-light: #4a6a8f;
--accent-subtle: rgba(45, 74, 111, 0.08);
```

**Never use:** Bright blues, purples, or gradients. This is not a SaaS dashboard.

### Status colors
```css
--success: #3d6b4f;  /* Muted forest green */
--warning: #8b6914;  /* Muted amber */
--error: #8b3d3d;    /* Muted burgundy */
```

## Spacing

Use the spacing scale consistently. Never use arbitrary values.

```css
--space-xs: 0.25rem;  /* 4px - tight gaps */
--space-sm: 0.5rem;   /* 8px - related elements */
--space-md: 1rem;     /* 16px - standard spacing */
--space-lg: 1.5rem;   /* 24px - section gaps */
--space-xl: 2rem;     /* 32px - major sections */
--space-2xl: 3rem;    /* 48px - page sections */
```

## Component patterns

### Cards / panels
- Background: `--bg-elevated` or `--bg-secondary`
- Border: `1px solid var(--border-subtle)`
- Border-radius: `6px` (small) or `8px` (large)
- Shadow: `--shadow-sm` (subtle, never dramatic)
- Padding: `--space-lg` or `--space-xl`

### Buttons
- Primary: `--accent` background, white text
- Hover: `--accent-light`
- Border-radius: `6px`
- Font: Serif, `font-weight: 500`
- Never use gradients or heavy shadows

### Form inputs
- Border: `1px solid var(--border-medium)`
- Focus: `border-color: var(--accent)` + subtle box-shadow
- Border-radius: `6px` - `8px`
- Placeholder: `--text-muted`, italic

### Links
- Color: `--accent`
- No underline by default
- Hover: subtle bottom border or color shift
- External links: add `↗` icon on hover

### Badges / tags
- Font: Mono, `0.75rem`, uppercase
- Background: semi-transparent accent or status color
- Border-radius: `4px`
- Padding: `--space-xs` vertical, `--space-sm` horizontal

## Motion and transitions

```css
--transition-fast: 150ms ease;   /* Hovers, small interactions */
--transition-medium: 250ms ease; /* Panel reveals */
--transition-slow: 400ms ease;   /* Page transitions */
```

### Animation principles
- **Restraint over flash** — subtle reveals, not dramatic animations
- **Staggered reveals** — source cards animate in sequence (50ms delay each)
- **Meaningful motion** — animation should guide attention, not distract
- Use `opacity` and `transform: translateY` for enters
- Never use bounce, elastic, or playful easing

## Visual details

### Paper texture
A subtle noise overlay creates warmth:
```css
body::before {
  background: url("data:image/svg+xml,...noise...");
  opacity: 0.015;
  pointer-events: none;
}
```

### Borders
```css
--border-subtle: rgba(26, 24, 20, 0.08);
--border-medium: rgba(26, 24, 20, 0.12);
--border-strong: rgba(26, 24, 20, 0.2);
```

### Shadows (subtle, warm-tinted)
```css
--shadow-sm: 0 1px 2px rgba(26, 24, 20, 0.04);
--shadow-md: 0 2px 8px rgba(26, 24, 20, 0.06);
--shadow-lg: 0 4px 16px rgba(26, 24, 20, 0.08);
```

## Content hierarchy examples

### Source card
```
┌─────────────────────────────────────────┐
│ ETHRESEARCH-TOPIC-1234 ↗         92.3% │  ← Mono, accent, similarity badge
│                                         │
│ The Shape of Issuance Curves to Come   │  ← Serif, primary, the "headline"
│ by vbuterin                            │  ← Italic, muted
│ ─────────────────────────────────────  │
│ Abstract > Introduction                │  ← Tertiary, separated
└─────────────────────────────────────────┘
```

### Response panel
```
┌─────────────────────────────────────────┐
│ RESPONSE                 claude · 1.2k │  ← Uppercase label + mono meta
│ ─────────────────────────────────────  │
│                                         │
│ ## Key features                         │  ← H2: border-bottom, primary
│                                         │
│ **Base fee mechanism**: EIP-1559...    │  ← Serif body, 500 for bold
│                                         │
│ ### Purpose and benefits               │  ← H3: accent color, no border
│                                         │
└─────────────────────────────────────────┘
```

## Do's and don'ts

### Do
- Use warm, muted colors
- Maintain generous whitespace
- Let typography do the heavy lifting
- Use subtle borders for separation
- Keep interactions quiet and refined

### Don't
- Use bright/saturated colors
- Add gratuitous animations
- Use heavy shadows or glows
- Mix too many font weights
- Make it look like a "tech product"

## Accessibility

- Maintain 4.5:1 contrast ratio minimum
- `:focus-visible` outlines on all interactive elements
- Semantic HTML (use `<button>`, `<label>`, proper headings)
- `aria-label` on icon-only buttons
- `aria-expanded` on expandable elements
- `role="status"` with `aria-live="polite"` for dynamic content

## File organization

```
app/
├── src/
│   ├── index.css      ← All styles (CSS variables at top)
│   ├── App.tsx        ← Main component
│   └── main.tsx       ← Entry point
├── DESIGN.md          ← This file
└── package.json
```

Keep styles in one file. The design system is small enough that splitting creates unnecessary complexity.
