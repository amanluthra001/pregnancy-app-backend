import mongoose from "mongoose";

const patientProfileSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, unique: true },
  age: Number,
  gestationalWeek: Number,
  gravida: Number,
  para: Number,
  contact: String,
  assignedDoctor: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
}, { timestamps: true });

export default mongoose.model("PatientProfile", patientProfileSchema);
