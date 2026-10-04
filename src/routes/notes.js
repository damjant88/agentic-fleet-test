const express = require('express');
const db = require('../db');
const { requireAuth } = require('../middleware/auth');
const { cleanText } = require('../validation');

const router = express.Router();
router.use(requireAuth);

router.get('/', (req, res) => {
  const notes = db
    .prepare('SELECT * FROM notes WHERE user_id = ? ORDER BY updated_at DESC')
    .all(req.session.userId);
  res.render('notes/index', { notes });
});

router.get('/new', (req, res) => {
  res.render('notes/form', { note: null, error: null });
});

router.post('/', (req, res) => {
  const title = cleanText(req.body.title, 200);
  const body = req.body.body ? cleanText(req.body.body, 20000) || '' : '';

  if (!title) {
    return res.status(400).render('notes/form', {
      note: { title: req.body.title, body: req.body.body },
      error: 'Title is required.',
    });
  }

  db.prepare('INSERT INTO notes (user_id, title, body) VALUES (?, ?, ?)').run(
    req.session.userId,
    title,
    body
  );

  res.redirect('/notes');
});

router.get('/:id/edit', (req, res) => {
  const note = db
    .prepare('SELECT * FROM notes WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.session.userId);
  if (!note) return res.status(404).render('errors/not-found');
  res.render('notes/form', { note, error: null });
});

router.post('/:id', (req, res) => {
  const existing = db
    .prepare('SELECT id FROM notes WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.session.userId);
  if (!existing) return res.status(404).render('errors/not-found');

  const title = cleanText(req.body.title, 200);
  const body = req.body.body ? cleanText(req.body.body, 20000) || '' : '';

  if (!title) {
    return res.status(400).render('notes/form', {
      note: { id: req.params.id, title: req.body.title, body: req.body.body },
      error: 'Title is required.',
    });
  }

  db.prepare(
    `UPDATE notes SET title = ?, body = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?`
  ).run(title, body, req.params.id, req.session.userId);

  res.redirect('/notes');
});

router.post('/:id/delete', (req, res) => {
  db.prepare('DELETE FROM notes WHERE id = ? AND user_id = ?').run(req.params.id, req.session.userId);
  res.redirect('/notes');
});

module.exports = router;
