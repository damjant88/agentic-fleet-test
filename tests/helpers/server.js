'use strict';
const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');

const SERVER_ENTRY = path.join(__dirname, '..', '..', 'src', 'server.js');

// Starts the app as a real child process (true black-box test of the server).
// Pass `dbPath` to reuse an existing SQLite file (e.g. to test restart persistence);
// otherwise a fresh temp file is created.
function startServer(port, dbPath) {
  const finalDbPath = dbPath || path.join(
    fs.mkdtempSync(path.join(os.tmpdir(), 'task-organizer-test-')),
    'app.db'
  );

  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [SERVER_ENTRY], {
      env: {
        ...process.env,
        PORT: String(port),
        DB_PATH: finalDbPath,
        SESSION_SECRET: 'test-secret',
        NODE_ENV: 'test',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let settled = false;
    let stderrBuf = '';

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        child.kill();
        reject(new Error(`Server on port ${port} did not start in time. stderr: ${stderrBuf}`));
      }
    }, 8000);

    child.stdout.on('data', (chunk) => {
      if (!settled && chunk.toString().includes('listening')) {
        settled = true;
        clearTimeout(timer);
        resolve({ child, dbPath: finalDbPath, baseUrl: `http://127.0.0.1:${port}` });
      }
    });
    child.stderr.on('data', (chunk) => {
      stderrBuf += chunk.toString();
    });
    child.on('error', (err) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(err);
      }
    });
    child.on('exit', (code) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(new Error(`Server on port ${port} exited early with code ${code}. stderr: ${stderrBuf}`));
      }
    });
  });
}

function stopServer(child) {
  return new Promise((resolve) => {
    if (child.exitCode !== null) return resolve();
    child.once('exit', () => resolve());
    child.kill();
  });
}

module.exports = { startServer, stopServer };
