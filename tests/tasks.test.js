'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { startServer, stopServer } = require('./helpers/server');
const { TestClient, uniqueEmail, signUp } = require('./helpers/client');
const { readDb } = require('./helpers/db');

const PORT = 3102;
let server;
let client;

before(async () => {
  server = await startServer(PORT);
  client = new TestClient(server.baseUrl);
  await signUp(client, uniqueEmail('tasks'), 'password123');
});

after(async () => {
  await stopServer(server.child);
});

function latestTaskId(title) {
  return readDb(server.dbPath, (db) =>
    db.prepare('SELECT id FROM tasks WHERE title = ? ORDER BY id DESC LIMIT 1').get(title).id
  );
}

test('a user can create a task', async () => {
  const title = 'Buy groceries ' + Date.now();
  const res = await client.post('/tasks', { title, description: 'milk, eggs' });
  assert.equal(res.status, 302);
  assert.equal(res.headers.get('location'), '/tasks');

  const list = await client.get('/tasks');
  const body = await list.text();
  assert.ok(body.includes(title), 'created task should appear in the list');
});

test('creating a task without a title fails', async () => {
  const res = await client.post('/tasks', { title: '   ' });
  assert.equal(res.status, 400);
});

test('a user can edit a task', async () => {
  const title = 'Old title ' + Date.now();
  await client.post('/tasks', { title });
  const id = latestTaskId(title);

  const newTitle = 'New title ' + Date.now();
  const res = await client.post(`/tasks/${id}`, { title: newTitle, description: 'updated' });
  assert.equal(res.status, 302);

  const list = await client.get('/tasks');
  const body = await list.text();
  assert.ok(body.includes(newTitle));
  assert.ok(!body.includes(title));
});

test('a user can delete a task', async () => {
  const title = 'To be deleted ' + Date.now();
  await client.post('/tasks', { title });
  const id = latestTaskId(title);

  const res = await client.post(`/tasks/${id}/delete`);
  assert.equal(res.status, 302);

  const row = readDb(server.dbPath, (db) => db.prepare('SELECT id FROM tasks WHERE id = ?').get(id));
  assert.equal(row, undefined, 'task row should be gone from the database');

  const list = await client.get('/tasks');
  const body = await list.text();
  assert.ok(!body.includes(title));
});

test('a user can toggle a task complete and incomplete', async () => {
  const title = 'Toggle me ' + Date.now();
  await client.post('/tasks', { title });
  const id = latestTaskId(title);

  let row = readDb(server.dbPath, (db) => db.prepare('SELECT completed FROM tasks WHERE id = ?').get(id));
  assert.equal(row.completed, 0);

  let res = await client.post(`/tasks/${id}/toggle`);
  assert.equal(res.status, 302);
  row = readDb(server.dbPath, (db) => db.prepare('SELECT completed FROM tasks WHERE id = ?').get(id));
  assert.equal(row.completed, 1, 'task should now be complete');

  res = await client.post(`/tasks/${id}/toggle`);
  assert.equal(res.status, 302);
  row = readDb(server.dbPath, (db) => db.prepare('SELECT completed FROM tasks WHERE id = ?').get(id));
  assert.equal(row.completed, 0, 'task should be back to incomplete');
});

test('completed filter reflects toggle state', async () => {
  const title = 'Filtered task ' + Date.now();
  await client.post('/tasks', { title });
  const id = latestTaskId(title);
  await client.post(`/tasks/${id}/toggle`);

  const completedList = await client.get('/tasks?filter=completed');
  assert.ok((await completedList.text()).includes(title));

  const activeList = await client.get('/tasks?filter=active');
  assert.ok(!(await activeList.text()).includes(title));
});
