'use strict';

// Minimal cookie-jar HTTP client so we can drive the app like a real browser
// (session cookie carried across requests) without pulling in supertest.
class TestClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.cookies = new Map();
  }

  _cookieHeader() {
    return [...this.cookies.entries()].map(([k, v]) => `${k}=${v}`).join('; ');
  }

  _absorbSetCookie(res) {
    const raw = typeof res.headers.getSetCookie === 'function' ? res.headers.getSetCookie() : [];
    for (const sc of raw) {
      const firstPart = sc.split(';')[0];
      const eq = firstPart.indexOf('=');
      const name = firstPart.slice(0, eq);
      const value = firstPart.slice(eq + 1);
      const isExpired = /expires=Thu, 01 Jan 1970/i.test(sc) || /Max-Age=0/i.test(sc);
      if (isExpired || value === '') {
        this.cookies.delete(name);
      } else {
        this.cookies.set(name, value);
      }
    }
  }

  async request(method, urlPath, { body, headers = {}, redirect = 'manual' } = {}) {
    const finalHeaders = { ...headers };
    const cookieHeader = this._cookieHeader();
    if (cookieHeader) finalHeaders['Cookie'] = cookieHeader;

    const init = { method, headers: finalHeaders, redirect };
    if (body !== undefined) {
      finalHeaders['Content-Type'] = 'application/x-www-form-urlencoded';
      init.body = new URLSearchParams(body).toString();
    }

    const res = await fetch(this.baseUrl + urlPath, init);
    this._absorbSetCookie(res);
    return res;
  }

  get(urlPath, opts) {
    return this.request('GET', urlPath, opts);
  }

  post(urlPath, body, opts) {
    return this.request('POST', urlPath, { ...opts, body });
  }

  hasSessionCookie() {
    return this.cookies.has('connect.sid');
  }

  clone() {
    const c = new TestClient(this.baseUrl);
    c.cookies = new Map(this.cookies);
    return c;
  }
}

function uniqueEmail(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}@example.com`;
}

async function signUp(client, email, password) {
  return client.post('/signup', { email, password });
}

async function logIn(client, email, password) {
  return client.post('/login', { email, password });
}

module.exports = { TestClient, uniqueEmail, signUp, logIn };
