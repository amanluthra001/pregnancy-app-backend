import express from 'express';
import QuestionnaireResponse from '../models/QuestionnaireResponse.js';
import requireAuth from '../middleware/auth.js';

const router = express.Router();

router.use(requireAuth('patient'));

router.post('/questionnaire', async (req, res) => {
    const { answers } = req.body;
    try {
        // Here you would call your ML service and get a result
        const mlPrediction = { riskLabel: 'Low', riskScore: 0.15 }; // Dummy data

        const newResponse = new QuestionnaireResponse({
            patient: req.user.id,
            answers: answers,
            riskLabel: mlPrediction.riskLabel,
            riskScore: mlPrediction.riskScore
        });
        await newResponse.save();
        res.status(201).json({ message: "Questionnaire submitted", result: newResponse });
    } catch (error) {
        res.status(500).json({ error: 'Server error submitting questionnaire.' });
    }
});

// Maternal Health Assessment endpoint
router.post('/maternal-assessment', async (req, res) => {
    const { answers } = req.body;
    try {
        // Call your Python ML service
        const mlResult = await callMLService('maternal', answers);
        
        // Save to database
        const newResponse = new QuestionnaireResponse({
            patient: req.user.id,
            answers: answers,
            riskLabel: mlResult.riskLabel,
            riskScore: mlResult.riskLevel,
            assessmentType: 'maternal'
        });
        await newResponse.save();

        res.status(201).json({ 
            message: "Maternal health assessment completed", 
            result: mlResult 
        });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'Failed to process maternal assessment.' });
    }
});

async function callMLService(modelType, answers) {
    try {
        const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:5001';
        const response = await fetch(`${mlServiceUrl}/predict`, {
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

export default router; // Use export default
