'use strict';
const Database = require('better-sqlite3');

// Opens a short-lived read connection to inspect state the HTTP API doesn't
// expose directly (e.g. verifying password hashing, or reading a row id).
function readDb(dbPath, fn) {
  const db = new Database(dbPath, { readonly: true, fileMustExist: true, timeout: 5000 });
  try {
    return fn(db);
  } finally {
    db.close();
  }
}

module.exports = { readDb };
