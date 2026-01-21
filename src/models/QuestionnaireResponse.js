import mongoose from "mongoose";

const questionnaireResponseSchema = new mongoose.Schema({
  patient: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  answers: { type: Object, required: true }, // Accept any shape for answers
  riskScore: { type: Number, min: 0, max: 1 },
  riskLabel: { type: String },
}, { timestamps: true });

export default mongoose.model("QuestionnaireResponse", questionnaireResponseSchema);
