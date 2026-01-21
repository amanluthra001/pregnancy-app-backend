import express from 'express';
import mongoose from 'mongoose';
import User from '../models/User.js';
import QuestionnaireResponse from '../models/QuestionnaireResponse.js';
import requireAuth from '../middleware/auth.js';

const router = express.Router();

router.use(requireAuth('doctor'));

router.post('/assign', async (req, res) => {
    const { patientId } = req.body;
    if (!mongoose.Types.ObjectId.isValid(patientId)) {
        return res.status(400).json({ error: 'Invalid Patient ID format.' });
    }
    try {
        const doctor = await User.findById(req.user.id);
        const patient = await User.findById(patientId);
        if (!patient || patient.role !== 'patient') {
            return res.status(404).json({ error: 'Patient not found.' });
        }
        if (doctor.assignedPatients.includes(patientId)) {
            return res.status(400).json({ error: 'Patient already assigned.' });
        }
        doctor.assignedPatients.push(patientId);
        await doctor.save();
        res.status(200).json({ message: 'Patient assigned successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Server error.' });
    }
});

router.get('/patients', async (req, res) => {
    try {
        const doctor = await User.findById(req.user.id).populate('assignedPatients', 'name email _id');
        res.status(200).json({ patients: doctor.assignedPatients });
    } catch (error) {
        res.status(500).json({ error: 'Server error.' });
    }
});

router.get('/patients/latest', async (req, res) => {
    try {
        const doctor = await User.findById(req.user.id);
        const questionnaires = await QuestionnaireResponse.find({ patient: { $in: doctor.assignedPatients } })
            .sort({ createdAt: -1 }).limit(20).populate('patient', 'name');
            const formatted = questionnaires.map(q => ({
                id: q._id,
                patientName: q.patient ? q.patient.name : 'Unknown',
                riskLabel: q.riskLabel,
                riskScore: q.riskScore,
                timestamp: q.createdAt
            }));
        res.status(200).json({ questionnaires: formatted });
    } catch (error) {
        res.status(500).json({ error: 'Server error.' });
    }
});

// Fetal Health Assessment endpoint
router.post('/fetal-assessment', async (req, res) => {
    const { answers } = req.body;
    try {
        // Call ML prediction service (will be set up separately)
        // For now, we'll use a placeholder that will call Python ML service
        const mlResult = await callMLService('fetal', answers);
        
        res.status(200).json({ 
            message: "Fetal health assessment completed", 
            result: mlResult 
        });
    } catch (error) {
        res.status(500).json({ error: 'Failed to process fetal assessment.' });
    }
});

async function callMLService(modelType, answers) {
    try {
        const response = await fetch('http://localhost:5001/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelType, data: answers })
        });
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('ML Service Error:', error);
        throw new Error('ML service unavailable');
    }
}

// Remove a patient from doctor's assigned list
router.delete('/patients/:patientId', async (req, res) => {
    const { patientId } = req.params;
    if (!mongoose.Types.ObjectId.isValid(patientId)) {
        return res.status(400).json({ error: 'Invalid Patient ID format.' });
    }
    try {
        const doctor = await User.findById(req.user.id);
        doctor.assignedPatients = doctor.assignedPatients.filter(id => id.toString() !== patientId);
        await doctor.save();
        res.status(200).json({ message: 'Patient removed.' });
    } catch (error) {
        res.status(500).json({ error: 'Server error.' });
    }
});

// Delete a questionnaire by ID
router.delete('/questionnaire/:id', async (req, res) => {
    const { id } = req.params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
        return res.status(400).json({ error: 'Invalid Questionnaire ID.' });
    }
    try {
        await QuestionnaireResponse.findByIdAndDelete(id);
        res.status(200).json({ message: 'Questionnaire deleted.' });
    } catch (error) {
        res.status(500).json({ error: 'Server error.' });
    }
});

export default router; // Use export default
