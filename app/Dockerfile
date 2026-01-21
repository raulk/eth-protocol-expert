FROM node:22-slim

# Enable corepack for pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml* ./

# Install dependencies
RUN pnpm install --frozen-lockfile || pnpm install

# Copy source
COPY . .

# Build
RUN pnpm build

EXPOSE 3000

# Use vite preview (serves built files with proxy from vite.config.ts)
CMD ["pnpm", "preview", "--host", "0.0.0.0", "--port", "3000"]
