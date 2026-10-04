'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { startServer, stopServer } = require('./helpers/server');
const { TestClient, uniqueEmail, signUp } = require('./helpers/client');
const { readDb } = require('./helpers/db');

const PORT = 3104;
let server;
let client;

before(async () => {
  server = await startServer(PORT);
  client = new TestClient(server.baseUrl);
  await signUp(client, uniqueEmail('reminders'), 'password123');
});

after(async () => {
  await stopServer(server.child);
});

test('a standalone reminder in the past is flagged Overdue', async () => {
  const title = 'Past reminder ' + Date.now();
  const res = await client.post('/reminders', { title, due_at: '2020-01-01T00:00' });
  assert.equal(res.status, 302);

  const list = await client.get('/reminders');
  const body = await list.text();
  const idx = body.indexOf(title);
  assert.notEqual(idx, -1, 'reminder should be listed');
  const nearby = body.slice(Math.max(0, idx - 400), idx + 400);
  assert.ok(nearby.includes('Overdue'), 'past reminder should be visually flagged as Overdue');
});

test('a standalone reminder far in the future is not flagged Overdue', async () => {
  const title = 'Future reminder ' + Date.now();
  const futureYear = new Date().getFullYear() + 5;
  await client.post('/reminders', { title, due_at: `${futureYear}-01-01T00:00` });

  const list = await client.get('/reminders');
  const body = await list.text();
  const idx = body.indexOf(title);
  assert.notEqual(idx, -1);
  const nearby = body.slice(Math.max(0, idx - 400), idx + 400);
  assert.ok(!nearby.includes('Overdue'), 'future reminder must not be flagged Overdue');
});

test('a task due date shows up in the combined reminders view, flagged when overdue', async () => {
  const title = 'Overdue task-reminder ' + Date.now();
  await client.post('/tasks', { title, due_at: '2020-06-15T09:00' });

  const list = await client.get('/reminders');
  const body = await list.text();
  const idx = body.indexOf(title);
  assert.notEqual(idx, -1, 'task with a due date should appear in the reminders view');
  const nearby = body.slice(Math.max(0, idx - 400), idx + 400);
  assert.ok(nearby.includes('Overdue'));
});

test('creating a reminder without a due date fails', async () => {
  const res = await client.post('/reminders', { title: 'No date here' });
  assert.equal(res.status, 400);
});

test('a completed task drops out of the reminders view', async () => {
  const title = 'Completable reminder task ' + Date.now();
  await client.post('/tasks', { title, due_at: '2020-01-01T00:00' });
  const id = readDb(server.dbPath, (db) =>
    db.prepare('SELECT id FROM tasks WHERE title = ? ORDER BY id DESC LIMIT 1').get(title).id
  );

  let reminders = await client.get('/reminders');
  assert.ok((await reminders.text()).includes(title), 'should be present before completion');

  await client.post(`/tasks/${id}/toggle`);

  reminders = await client.get('/reminders');
  assert.ok(!(await reminders.text()).includes(title), 'completed task should drop out of reminders view');
});
