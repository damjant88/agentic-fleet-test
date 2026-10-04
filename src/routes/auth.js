const express = require('express');
const bcrypt = require('bcrypt');
const db = require('../db');
const { isValidEmail, isValidPassword } = require('../validation');

const router = express.Router();
const SALT_ROUNDS = 12;

router.get('/signup', (req, res) => {
  if (req.session.userId) return res.redirect('/tasks');
  res.render('auth/signup', { error: null, email: '' });
});

router.post('/signup', async (req, res) => {
  const { email, password } = req.body;

  if (!isValidEmail(email)) {
    return res.status(400).render('auth/signup', { error: 'Please enter a valid email address.', email: email || '' });
  }
  if (!isValidPassword(password)) {
    return res.status(400).render('auth/signup', { error: 'Password must be at least 8 characters.', email: email || '' });
  }

  const existing = db.prepare('SELECT id FROM users WHERE email = ?').get(email);
  if (existing) {
    return res.status(400).render('auth/signup', { error: 'An account with that email already exists.', email });
  }

  const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);
  const result = db
    .prepare('INSERT INTO users (email, password_hash) VALUES (?, ?)')
    .run(email, passwordHash);

  req.session.regenerate((err) => {
    if (err) return res.status(500).render('auth/signup', { error: 'Something went wrong. Please try again.', email });
    req.session.userId = result.lastInsertRowid;
    req.session.email = email;
    res.redirect('/tasks');
  });
});

router.get('/login', (req, res) => {
  if (req.session.userId) return res.redirect('/tasks');
  res.render('auth/login', { error: null, email: '' });
});

router.post('/login', async (req, res) => {
  const { email, password } = req.body;

  if (!isValidEmail(email) || typeof password !== 'string' || password.length === 0) {
    return res.status(400).render('auth/login', { error: 'Invalid email or password.', email: email || '' });
  }

  const user = db.prepare('SELECT id, password_hash FROM users WHERE email = ?').get(email);
  if (!user) {
    return res.status(401).render('auth/login', { error: 'Invalid email or password.', email });
  }

  const match = await bcrypt.compare(password, user.password_hash);
  if (!match) {
    return res.status(401).render('auth/login', { error: 'Invalid email or password.', email });
  }

  req.session.regenerate((err) => {
    if (err) return res.status(500).render('auth/login', { error: 'Something went wrong. Please try again.', email });
    req.session.userId = user.id;
    req.session.email = email;
    res.redirect('/tasks');
  });
});

router.post('/logout', (req, res) => {
  req.session.destroy(() => {
    res.clearCookie('connect.sid');
    res.redirect('/login');
  });
});

module.exports = router;
