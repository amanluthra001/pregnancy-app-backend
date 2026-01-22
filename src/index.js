import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";

import authRoutes from "./routes/auth.js";
import patientRoutes from "./routes/patient.js";
import doctorRoutes from "./routes/doctor.js";

dotenv.config();
const app = express();

/* ---------------- middleware ---------------- */
app.use(cors({
  origin: [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://fetalhealth-three.vercel.app"
  ],
  credentials: true
}));
app.use(express.json());
app.use(cookieParser());

/* ---------------- routes ---------------- */
app.get("/api/health", (_req, res) => res.json({ ok: true }));
app.use("/api/auth", authRoutes);
app.use("/api/patient", patientRoutes);
app.use("/api/doctor", doctorRoutes);

/* ---------------- database ---------------- */
const MONGO_URI = process.env.MONGO_URI;

mongoose.connect(MONGO_URI)
  .then(() => console.log("Connected to MongoDB"))
  .catch(err => console.error("MongoDB error:", err));

/* ---------------- LISTEN (Render needs this) ---------------- */
const PORT = process.env.PORT || 3000;

// Only listen when NOT running on Vercel
if (process.env.RENDER || process.env.NODE_ENV !== "production") {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}

export default app;
