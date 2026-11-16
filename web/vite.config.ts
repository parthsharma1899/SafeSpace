import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" keeps asset paths relative so the built playground works on GitHub Pages.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "dist" },
});
