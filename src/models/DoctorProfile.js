import mongoose from "mongoose";

const doctorProfileSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, unique: true },
  specialization: String,
  hospital: String,
  contact: String,
}, { timestamps: true });

export default mongoose.model("DoctorProfile", doctorProfileSchema);
