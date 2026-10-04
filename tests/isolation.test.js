'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { startServer, stopServer } = require('./helpers/server');
const { TestClient, uniqueEmail, signUp } = require('./helpers/client');
const { readDb } = require('./helpers/db');

const PORT = 3105;
let server;

before(async () => {
  server = await startServer(PORT);
});

after(async () => {
  await stopServer(server.child);
});

test('two users cannot see each other\'s tasks, notes, or reminders', async () => {
  const alice = new TestClient(server.baseUrl);
  await signUp(alice, uniqueEmail('alice'), 'password123');
  const taskTitle = 'Alice private task ' + Date.now();
  const noteTitle = 'Alice private note ' + Date.now();
  const reminderTitle = 'Alice private reminder ' + Date.now();
  await alice.post('/tasks', { title: taskTitle });
  await alice.post('/notes', { title: noteTitle, body: 'secret' });
  await alice.post('/reminders', { title: reminderTitle, due_at: '2030-01-01T00:00' });

  const bob = new TestClient(server.baseUrl);
  await signUp(bob, uniqueEmail('bob'), 'password123');

  const [tasksBody, notesBody, remindersBody] = await Promise.all([
    bob.get('/tasks').then((r) => r.text()),
    bob.get('/notes').then((r) => r.text()),
    bob.get('/reminders').then((r) => r.text()),
  ]);

  assert.ok(!tasksBody.includes(taskTitle), "bob must not see alice's task");
  assert.ok(tasksBody.includes('No tasks yet'), "bob's task list should be empty");
  assert.ok(!notesBody.includes(noteTitle), "bob must not see alice's note");
  assert.ok(notesBody.includes('No notes yet'), "bob's note list should be empty");
  assert.ok(!remindersBody.includes(reminderTitle), "bob must not see alice's reminder");
});

test('a user cannot access another user\'s task/note/reminder by guessing the id', async () => {
  const alice = new TestClient(server.baseUrl);
  await signUp(alice, uniqueEmail('alice-id'), 'password123');
  const taskTitle = 'Alice guessable task ' + Date.now();
  const noteTitle = 'Alice guessable note ' + Date.now();
  await alice.post('/tasks', { title: taskTitle });
  await alice.post('/notes', { title: noteTitle, body: 'x' });

  const taskId = readDb(server.dbPath, (db) =>
    db.prepare('SELECT id FROM tasks WHERE title = ?').get(taskTitle).id
  );
  const noteId = readDb(server.dbPath, (db) =>
    db.prepare('SELECT id FROM notes WHERE title = ?').get(noteTitle).id
  );

  const bob = new TestClient(server.baseUrl);
  await signUp(bob, uniqueEmail('bob-id'), 'password123');

  const editTaskRes = await bob.get(`/tasks/${taskId}/edit`);
  assert.equal(editTaskRes.status, 404, "bob must not be able to open alice's task edit form");

  const editNoteRes = await bob.get(`/notes/${noteId}/edit`);
  assert.equal(editNoteRes.status, 404, "bob must not be able to open alice's note edit form");

  await bob.post(`/tasks/${taskId}/delete`);
  const stillThere = readDb(server.dbPath, (db) => db.prepare('SELECT id FROM tasks WHERE id = ?').get(taskId));
  assert.ok(stillThere, "bob's delete attempt on alice's task must be a no-op");

  await bob.post(`/tasks/${taskId}/toggle`);
  const unchanged = readDb(server.dbPath, (db) =>
    db.prepare('SELECT completed FROM tasks WHERE id = ?').get(taskId)
  );
  assert.equal(unchanged.completed, 0, "bob's toggle attempt on alice's task must be a no-op");
});
