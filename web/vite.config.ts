import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The app calls the API at /api; in development and preview Vite forwards that to
// the backend, so the browser sees one origin. Override with VITE_API_URL.
const api = {
  "/api": {
    target: process.env.SAFESPACE_API ?? "http://127.0.0.1:8080",
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { proxy: api },
  preview: { proxy: api },
  build: { outDir: "dist" },
});
