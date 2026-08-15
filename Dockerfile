FROM node:20-bookworm

# Install ffmpeg and canvas native dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev pkg-config python3 libpng-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY package.json bun.lock* package-lock.json* ./

# Install dependencies including canvas native compilation, then wipe npm cache
# (single layer so node_modules is only snapshotted once in overlayfs)
RUN npm ci && npm cache clean --force

# Copy the rest of the application
COPY . .

# Build the frontend and backend (Vite + esbuild)
RUN npm run build

# Expose the standard port
EXPOSE 8080

# The server listens on PORT env var if present, otherwise 3000.
# We'll set it to 8080 which is common for Google Cloud Run / Compute Engine
ENV PORT=8080
ENV NODE_ENV=production

# Start the built API server
CMD ["node", "dist/server.cjs"]
