import express from 'express';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import User from '../models/User.js';

const router = express.Router();

// POST /api/auth/register
router.post('/register', async (req, res) => {
  const { name, email, password, role } = req.body;
  if (!name || !email || !password || !role) {
    return res.status(400).json({ error: 'All fields are required.' });
  }
  try {
    const existing = await User.findOne({ email });
    if (existing) {
      return res.status(409).json({ error: 'Email already registered.' });
    }
    const passwordHash = await bcrypt.hash(password, 10);
    const user = await User.create({ name, email, passwordHash, role });
    // Do not log in after registration, just return user info
    res.status(201).json({ message: 'Registration successful', user: { _id: user._id, name: user.name, email: user.email, role: user.role } });
  } catch (err) {
    res.status(500).json({ error: 'Server error during registration.' });
  }
});

// POST /api/auth/login
router.post('/login', async (req, res) => {
  const { email, password, role } = req.body;
  if (!email || !password || !role) {
    return res.status(400).json({ error: 'Email, password, and role are required.' });
  }
  try {
    const user = await User.findOne({ email, role });
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials.' });
    }
    const valid = await bcrypt.compare(password, user.passwordHash);
    if (!valid) {
      return res.status(401).json({ error: 'Invalid credentials.' });
    }
    const token = jwt.sign({ id: user._id, name: user.name, role: user.role }, process.env.JWT_SECRET || 'your_default_secret', { expiresIn: '7d' });
    res.json({ token, user: { _id: user._id, name: user.name, email: user.email, role: user.role } });
  } catch (err) {
    res.status(500).json({ error: 'Server error during login.' });
  }
});

export default router;
