'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { startServer, stopServer } = require('./helpers/server');
const { TestClient, uniqueEmail, signUp } = require('./helpers/client');
const { readDb } = require('./helpers/db');

const PORT = 3103;
let server;
let client;

before(async () => {
  server = await startServer(PORT);
  client = new TestClient(server.baseUrl);
  await signUp(client, uniqueEmail('notes'), 'password123');
});

after(async () => {
  await stopServer(server.child);
});

function latestNoteId(title) {
  return readDb(server.dbPath, (db) =>
    db.prepare('SELECT id FROM notes WHERE title = ? ORDER BY id DESC LIMIT 1').get(title).id
  );
}

test('a user can create a note', async () => {
  const title = 'Shopping list ' + Date.now();
  const res = await client.post('/notes', { title, body: 'bread, cheese' });
  assert.equal(res.status, 302);
  assert.equal(res.headers.get('location'), '/notes');

  const list = await client.get('/notes');
  const body = await list.text();
  assert.ok(body.includes(title));
});

test('creating a note without a title fails', async () => {
  const res = await client.post('/notes', { title: '', body: 'no title here' });
  assert.equal(res.status, 400);
});

test('a user can edit a note', async () => {
  const title = 'Draft note ' + Date.now();
  await client.post('/notes', { title, body: 'v1' });
  const id = latestNoteId(title);

  const newTitle = 'Final note ' + Date.now();
  const res = await client.post(`/notes/${id}`, { title: newTitle, body: 'v2' });
  assert.equal(res.status, 302);

  const row = readDb(server.dbPath, (db) => db.prepare('SELECT title, body FROM notes WHERE id = ?').get(id));
  assert.equal(row.title, newTitle);
  assert.equal(row.body, 'v2');
});

test('a user can delete a note', async () => {
  const title = 'Temporary note ' + Date.now();
  await client.post('/notes', { title, body: 'x' });
  const id = latestNoteId(title);

  const res = await client.post(`/notes/${id}/delete`);
  assert.equal(res.status, 302);

  const row = readDb(server.dbPath, (db) => db.prepare('SELECT id FROM notes WHERE id = ?').get(id));
  assert.equal(row, undefined);
});

test('note body is HTML-escaped on render (no stored XSS)', async () => {
  const title = 'XSS check ' + Date.now();
  const payload = '<script>alert(1)</script>';
  await client.post('/notes', { title, body: payload });

  const list = await client.get('/notes');
  const bodyHtml = await list.text();
  assert.ok(!bodyHtml.includes('<script>alert(1)</script>'), 'raw script tag must not appear unescaped');
  assert.ok(bodyHtml.includes('&lt;script&gt;'), 'expected the payload to be HTML-escaped');
});
