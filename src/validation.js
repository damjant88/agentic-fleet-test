const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function isValidEmail(email) {
  return typeof email === 'string' && email.length <= 254 && EMAIL_RE.test(email);
}

function isValidPassword(password) {
  return typeof password === 'string' && password.length >= 8 && password.length <= 200;
}

function cleanText(value, maxLen) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (trimmed.length === 0 || trimmed.length > maxLen) return null;
  return trimmed;
}

function isValidDueAt(value) {
  if (value === undefined || value === null || value === '') return true;
  if (typeof value !== 'string') return false;
  const d = new Date(value);
  return !Number.isNaN(d.getTime());
}

module.exports = { isValidEmail, isValidPassword, cleanText, isValidDueAt };
