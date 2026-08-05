import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    // Allow the app to be served through the hosted preview proxy.
    allowedHosts: true,
    hmr: {
      // The preview is served over HTTPS through a proxy, so the HMR
      // websocket must connect back via wss on the standard TLS port
      // instead of ws://host:<dev-port>, which the proxy cannot route.
      protocol: 'wss',
      clientPort: 443,
    },
  },
});
