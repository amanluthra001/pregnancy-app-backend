import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";

// Import all routes using ES Module syntax
import authRoutes from "./routes/auth.js";
import patientRoutes from "./routes/patient.js";
import doctorRoutes from "./routes/doctor.js";

dotenv.config();
const app = express();

// Middleware
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

// Serve static files from public directory
// app.use(express.static('./public'));

// Serve index.html for root path
// app.get('/', (req, res) => {
//   res.sendFile('index.html', { root: './public' });
// });

// API Routes
app.get("/api/health", (_req, res) => res.json({ ok: true }));
app.use("/api/auth", authRoutes);
app.use("/api/patient", patientRoutes);
app.use("/api/doctor", doctorRoutes);

// Database Connection
const MONGO_URI = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/pregnancy_app";

mongoose.connect(MONGO_URI).then(() => {
  console.log("Connected to MongoDB");
}).catch(err => {
  console.error("Failed to connect to MongoDB", err);
});

// Export the app for Vercel serverless functions
export default app;
