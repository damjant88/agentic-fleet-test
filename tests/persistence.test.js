'use strict';
const { test, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { startServer, stopServer } = require('./helpers/server');
const { TestClient, uniqueEmail, signUp } = require('./helpers/client');

const PORT = 3106;
let cleanupDir;

after(() => {
  if (cleanupDir) fs.rmSync(cleanupDir, { recursive: true, force: true });
});

test('data persists across a server restart', async () => {
  cleanupDir = fs.mkdtempSync(path.join(os.tmpdir(), 'task-organizer-persist-'));
  const dbPath = path.join(cleanupDir, 'app.db');

  const first = await startServer(PORT, dbPath);
  const client = new TestClient(first.baseUrl);
  const email = uniqueEmail('persist');
  const password = 'password123';
  await signUp(client, email, password);

  const taskTitle = 'Survives restart ' + Date.now();
  await client.post('/tasks', { title: taskTitle });

  const beforeRestart = await client.get('/tasks');
  assert.ok((await beforeRestart.text()).includes(taskTitle));

  await stopServer(first.child);

  const second = await startServer(PORT, dbPath);
  try {
    const freshClient = new TestClient(second.baseUrl);
    const loginRes = await freshClient.post('/login', { email, password });
    assert.equal(loginRes.status, 302, 'should be able to log back in with the same credentials after restart');

    const afterRestart = await freshClient.get('/tasks');
    assert.equal(afterRestart.status, 200);
    const body = await afterRestart.text();
    assert.ok(body.includes(taskTitle), 'task created before restart should still be present');
  } finally {
    await stopServer(second.child);
  }
});
