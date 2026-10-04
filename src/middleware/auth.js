function requireAuth(req, res, next) {
  if (!req.session || !req.session.userId) {
    if (req.accepts('html')) {
      return res.redirect('/login');
    }
    return res.status(401).json({ error: 'Authentication required' });
  }
  next();
}

module.exports = { requireAuth };
