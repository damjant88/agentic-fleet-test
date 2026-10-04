const express = require('express');
const db = require('../db');
const { requireAuth } = require('../middleware/auth');
const { cleanText, isValidDueAt } = require('../validation');

const router = express.Router();
router.use(requireAuth);

function normalizeDueAt(dueAt) {
  return dueAt && dueAt.trim() !== '' ? new Date(dueAt).toISOString() : null;
}

router.get('/', (req, res) => {
  const filter = ['active', 'completed'].includes(req.query.filter) ? req.query.filter : 'all';

  let sql = 'SELECT * FROM tasks WHERE user_id = ?';
  if (filter === 'active') sql += ' AND completed = 0';
  if (filter === 'completed') sql += ' AND completed = 1';
  sql += ' ORDER BY (due_at IS NULL), due_at ASC, created_at DESC';

  const tasks = db.prepare(sql).all(req.session.userId);
  res.render('tasks/index', { tasks, filter, now: new Date().toISOString() });
});

router.get('/new', (req, res) => {
  res.render('tasks/form', { task: null, error: null });
});

router.post('/', (req, res) => {
  const title = cleanText(req.body.title, 200);
  const description = req.body.description ? cleanText(req.body.description, 5000) || '' : '';
  const dueAtRaw = req.body.due_at;

  if (!title || !isValidDueAt(dueAtRaw)) {
    return res.status(400).render('tasks/form', {
      task: { title: req.body.title, description: req.body.description, due_at: dueAtRaw },
      error: 'Title is required and due date must be valid.',
    });
  }

  db.prepare(
    'INSERT INTO tasks (user_id, title, description, due_at) VALUES (?, ?, ?, ?)'
  ).run(req.session.userId, title, description, normalizeDueAt(dueAtRaw));

  res.redirect('/tasks');
});

router.get('/:id/edit', (req, res) => {
  const task = db
    .prepare('SELECT * FROM tasks WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.session.userId);
  if (!task) return res.status(404).render('errors/not-found');
  res.render('tasks/form', { task, error: null });
});

router.post('/:id', (req, res) => {
  const existing = db
    .prepare('SELECT id FROM tasks WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.session.userId);
  if (!existing) return res.status(404).render('errors/not-found');

  const title = cleanText(req.body.title, 200);
  const description = req.body.description ? cleanText(req.body.description, 5000) || '' : '';
  const dueAtRaw = req.body.due_at;

  if (!title || !isValidDueAt(dueAtRaw)) {
    return res.status(400).render('tasks/form', {
      task: { id: req.params.id, title: req.body.title, description: req.body.description, due_at: dueAtRaw },
      error: 'Title is required and due date must be valid.',
    });
  }

  db.prepare(
    `UPDATE tasks SET title = ?, description = ?, due_at = ?, updated_at = datetime('now')
     WHERE id = ? AND user_id = ?`
  ).run(title, description, normalizeDueAt(dueAtRaw), req.params.id, req.session.userId);

  res.redirect('/tasks');
});

router.post('/:id/toggle', (req, res) => {
  const task = db
    .prepare('SELECT completed FROM tasks WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.session.userId);
  if (!task) return res.status(404).render('errors/not-found');

  db.prepare(
    `UPDATE tasks SET completed = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?`
  ).run(task.completed ? 0 : 1, req.params.id, req.session.userId);

  res.redirect('/tasks');
});

router.post('/:id/delete', (req, res) => {
  db.prepare('DELETE FROM tasks WHERE id = ? AND user_id = ?').run(req.params.id, req.session.userId);
  res.redirect('/tasks');
});

module.exports = router;
