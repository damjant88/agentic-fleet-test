const path = require('path');
const express = require('express');
const session = require('express-session');

require('./db');

const authRoutes = require('./routes/auth');
const taskRoutes = require('./routes/tasks');
const noteRoutes = require('./routes/notes');
const reminderRoutes = require('./routes/reminders');

const app = express();
const isProduction = process.env.NODE_ENV === 'production';
const PORT = process.env.PORT || 3000;

if (isProduction) {
  app.set('trust proxy', 1);
}

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

app.use(express.urlencoded({ extended: false }));
app.use(express.static(path.join(__dirname, 'public')));

app.use(
  session({
    name: 'connect.sid',
    secret: process.env.SESSION_SECRET || 'dev-secret-change-me',
    resave: false,
    saveUninitialized: false,
    cookie: {
      httpOnly: true,
      sameSite: 'lax',
      secure: isProduction,
      maxAge: 1000 * 60 * 60 * 24 * 7,
    },
  })
);

app.use((req, res, next) => {
  res.locals.userEmail = req.session.email || null;
  next();
});

app.get('/', (req, res) => {
  res.redirect(req.session.userId ? '/tasks' : '/login');
});

app.use('/', authRoutes);
app.use('/tasks', taskRoutes);
app.use('/notes', noteRoutes);
app.use('/reminders', reminderRoutes);

app.use((req, res) => {
  res.status(404).render('errors/not-found');
});

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).send('Something went wrong.');
});

app.listen(PORT, () => {
  console.log(`Task Organizer listening on http://localhost:${PORT}`);
});
