const express = require('express');
const db = require('../db');
const { requireAuth } = require('../middleware/auth');
const { cleanText, isValidDueAt } = require('../validation');

const router = express.Router();
router.use(requireAuth);

function normalizeDueAt(dueAt) {
  return new Date(dueAt).toISOString();
}

function getCombinedReminders(userId) {
  const standalone = db
    .prepare('SELECT id, title, due_at, \'reminder\' AS kind FROM reminders WHERE user_id = ?')
    .all(userId);
  const taskReminders = db
    .prepare(
      "SELECT id, title, due_at, 'task' AS kind FROM tasks WHERE user_id = ? AND due_at IS NOT NULL AND completed = 0"
    )
    .all(userId);

  return [...standalone, ...taskReminders].sort((a, b) => new Date(a.due_at) - new Date(b.due_at));
}

router.get('/', (req, res) => {
  const items = getCombinedReminders(req.session.userId);
  res.render('reminders/index', { items, now: new Date().toISOString() });
});

router.get('/new', (req, res) => {
  res.render('reminders/form', { reminder: null, error: null });
});

router.post('/', (req, res) => {
  const title = cleanText(req.body.title, 200);
  const dueAtRaw = req.body.due_at;

  if (!title || !dueAtRaw || !isValidDueAt(dueAtRaw)) {
    return res.status(400).render('reminders/form', {
      reminder: { title: req.body.title, due_at: dueAtRaw },
      error: 'Title and a valid due date/time are required.',
    });
  }

  db.prepare('INSERT INTO reminders (user_id, title, due_at) VALUES (?, ?, ?)').run(
    req.session.userId,
    title,
    normalizeDueAt(dueAtRaw)
  );

  res.redirect('/reminders');
});

router.get('/:id/edit', (req, res) => {
  const reminder = db
    .prepare('SELECT * FROM reminders WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.session.userId);
  if (!reminder) return res.status(404).render('errors/not-found');
  res.render('reminders/form', { reminder, error: null });
});

router.post('/:id', (req, res) => {
  const existing = db
    .prepare('SELECT id FROM reminders WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.session.userId);
  if (!existing) return res.status(404).render('errors/not-found');

  const title = cleanText(req.body.title, 200);
  const dueAtRaw = req.body.due_at;

  if (!title || !dueAtRaw || !isValidDueAt(dueAtRaw)) {
    return res.status(400).render('reminders/form', {
      reminder: { id: req.params.id, title: req.body.title, due_at: dueAtRaw },
      error: 'Title and a valid due date/time are required.',
    });
  }

  db.prepare(
    `UPDATE reminders SET title = ?, due_at = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?`
  ).run(title, normalizeDueAt(dueAtRaw), req.params.id, req.session.userId);

  res.redirect('/reminders');
});

router.post('/:id/delete', (req, res) => {
  db.prepare('DELETE FROM reminders WHERE id = ? AND user_id = ?').run(req.params.id, req.session.userId);
  res.redirect('/reminders');
});

module.exports = router;
