import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      gsplat: path.resolve(__dirname, "../gsplat.js/dist/index.es.js"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: 9090,
  },
  // Serve .ply from the public dir
  publicDir: "public",
});
