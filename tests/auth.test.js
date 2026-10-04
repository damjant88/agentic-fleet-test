'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { startServer, stopServer } = require('./helpers/server');
const { TestClient, uniqueEmail, signUp, logIn } = require('./helpers/client');
const { readDb } = require('./helpers/db');

const PORT = 3101;
let server;

before(async () => {
  server = await startServer(PORT);
});

after(async () => {
  await stopServer(server.child);
});

test('signup creates a user and starts an authenticated session', async () => {
  const client = new TestClient(server.baseUrl);
  const email = uniqueEmail('signup');
  const res = await signUp(client, email, 'correct-horse');

  assert.equal(res.status, 302);
  assert.equal(res.headers.get('location'), '/tasks');
  assert.ok(client.hasSessionCookie(), 'expected a session cookie after signup');

  const row = readDb(server.dbPath, (db) => db.prepare('SELECT * FROM users WHERE email = ?').get(email));
  assert.ok(row, 'user row should exist in the database');
});

test('passwords are stored hashed, never in plaintext', async () => {
  const client = new TestClient(server.baseUrl);
  const email = uniqueEmail('hash');
  const password = 'super-secret-1';
  await signUp(client, email, password);

  const row = readDb(server.dbPath, (db) => db.prepare('SELECT * FROM users WHERE email = ?').get(email));
  assert.ok(row, 'user row should exist');
  assert.notEqual(row.password_hash, password, 'password_hash must not equal the plaintext password');
  assert.match(row.password_hash, /^\$2[aby]\$/, 'expected a bcrypt hash format');
  assert.equal(row.password_hash.includes(password), false, 'plaintext password leaked into stored hash');
});

test('signup rejects a duplicate email', async () => {
  const client = new TestClient(server.baseUrl);
  const email = uniqueEmail('dup');
  await signUp(client, email, 'first-password');

  const client2 = new TestClient(server.baseUrl);
  const res = await signUp(client2, email, 'second-password');
  assert.equal(res.status, 400);
  assert.equal(client2.hasSessionCookie(), false);
});

test('login succeeds with correct credentials', async () => {
  const signupClient = new TestClient(server.baseUrl);
  const email = uniqueEmail('login-ok');
  const password = 'right-password-1';
  await signUp(signupClient, email, password);
  await signupClient.post('/logout');

  const loginClient = new TestClient(server.baseUrl);
  const res = await logIn(loginClient, email, password);
  assert.equal(res.status, 302);
  assert.equal(res.headers.get('location'), '/tasks');
  assert.ok(loginClient.hasSessionCookie());

  const tasksRes = await loginClient.get('/tasks');
  assert.equal(tasksRes.status, 200);
});

test('login fails with incorrect password', async () => {
  const signupClient = new TestClient(server.baseUrl);
  const email = uniqueEmail('login-bad');
  await signUp(signupClient, email, 'the-real-password');
  await signupClient.post('/logout');

  const loginClient = new TestClient(server.baseUrl);
  const res = await logIn(loginClient, email, 'totally-wrong-password');
  assert.equal(res.status, 401);
  assert.equal(loginClient.hasSessionCookie(), false, 'a failed login must not grant a session');

  const tasksRes = await loginClient.get('/tasks');
  assert.equal(tasksRes.status, 302);
  assert.equal(tasksRes.headers.get('location'), '/login');
});

test('login fails for an email that does not exist', async () => {
  const client = new TestClient(server.baseUrl);
  const res = await logIn(client, uniqueEmail('nobody'), 'whatever-password');
  assert.equal(res.status, 401);
});

test('logout invalidates the session server-side', async () => {
  const client = new TestClient(server.baseUrl);
  const email = uniqueEmail('logout');
  await signUp(client, email, 'a-password-1');

  let res = await client.get('/tasks');
  assert.equal(res.status, 200, 'should be authenticated right after signup');

  res = await client.post('/logout');
  assert.equal(res.status, 302);
  assert.equal(res.headers.get('location'), '/login');

  res = await client.get('/tasks');
  assert.equal(res.status, 302, 'subsequent authenticated request must fail after logout');
  assert.equal(res.headers.get('location'), '/login');
});

test('unauthenticated requests to protected routes are rejected server-side', async () => {
  const anon = new TestClient(server.baseUrl);

  const htmlRes = await anon.get('/tasks');
  assert.equal(htmlRes.status, 302);
  assert.equal(htmlRes.headers.get('location'), '/login');

  const jsonRes = await anon.get('/tasks', { headers: { Accept: 'application/json' } });
  assert.equal(jsonRes.status, 401);

  const notesRes = await anon.get('/notes');
  assert.equal(notesRes.status, 302);

  const remindersRes = await anon.get('/reminders');
  assert.equal(remindersRes.status, 302);

  const postRes = await anon.post('/tasks', { title: 'sneaky task' });
  assert.equal(postRes.status, 302);
  assert.equal(postRes.headers.get('location'), '/login');
});
