# PRD: Personal Task Organizer (Tasks, Notes & Reminders)

## 1. Overview

**Purpose.** This app is the target build for testing an agentic QE fleet (a swarm
of Claude Flow agents implementing and validating a small app end-to-end). It is a
**personal task organizer** combining tasks, notes, and reminders behind a
login-authenticated, single-user-per-account experience.

**Source spec.** README.md (2 lines):
> Testing and running agentic qe fleet by making a small custom app with login
> authentication. App should serve as a personal task organizer, notes and reminders.

**Guiding principle.** Minimum viable scope, one process, no build step, nothing
that isn't required to satisfy the README. This keeps the implementation surface
small enough for an agent swarm to build and verify in a single pass.

## 2. Goals / Non-Goals

**Goals**
- A user can sign up, log in, and log out.
- A logged-in user can manage their own tasks, notes, and reminders (CRUD).
- Reminders surface visually when due/overdue.
- Each user's data is private to that user.

**Non-Goals (explicitly out of scope for MVP)**
- Password reset via email, email verification, OAuth/SSO, MFA
- Push/email/SMS delivery of reminders (in-app display only)
- Sharing, collaboration, or multi-user assignment
- Native mobile apps (responsive web only)
- Third-party integrations (calendar sync, Slack, etc.)
- Rich text/markdown editing, file attachments
- Tagging, categories, search/filter, sort options beyond due date
- Recurring reminders, offline support/sync
- Theming/dark mode, WCAG AA accessibility audit
- Admin panel, analytics, audit logs, rate-limiting infra, CAPTCHA

## 3. User Stories

1. As a user, I want to sign up with email/password so that I can create a private account.
2. As a user, I want to log in and log out so that my data stays secure between sessions.
3. As a user, I want to create, edit, and delete tasks so that I can track things I need to do.
4. As a user, I want to mark a task complete so that I can see my progress and focus on what's left.
5. As a user, I want to write and edit freeform notes so that I can capture information outside my task list.
6. As a user, I want to set a due date/reminder on a task so that I know what's coming up and don't miss it.

## 4. Feature Scope

### 4.1 Auth
- Sign up with email + password (hashed storage).
- Log in / log out.
- Session persists across page reloads until logout or expiry.

### 4.2 Tasks
- Create, read, update, delete.
- Mark complete/incomplete.
- Optional due date.
- List view with filter: all / active / completed.

### 4.3 Notes
- Create, read, update, delete.
- Title + plain-text body.
- List view.

### 4.4 Reminders
- Due date/time attached to a task, or a standalone reminder item.
- List sorted by soonest due date.
- Visual flag ("due today" / "overdue") in-app; no external delivery (no email/push/SMS).

### 4.5 Cross-cutting
- Per-user data isolation — a user only ever sees their own tasks/notes/reminders.
- Simple nav between Tasks / Notes / Reminders.
- Basic responsive layout (desktop + mobile browser widths); list + detail/edit view per entity.

## 5. Technical Architecture (recommended)

| Concern | Choice | Rationale |
|---|---|---|
| Shape | Single server-rendered Node.js app (Express + EJS templates), light `fetch`-based JS for interactivity | One process, one deploy, no bundler/build step, no CORS/API-contract surface to get wrong |
| Backend framework | Express | Most common, best-documented — lowest hallucination risk for agentic implementation |
| Auth | Session cookies (`express-session`) + `bcrypt` password hashing | No token refresh/client-storage complexity; browser handles the cookie |
| Storage | SQLite via `better-sqlite3` | Real, relational, file-based persistence with zero external service to run |
| Reminders delivery | In-app/UI-only (badges on load) | Satisfies README's "reminders" requirement without notification infrastructure |

Overall bias: one process, one language, one datastore, no build step — everything
an agent swarm can implement, run, and verify in a single short pass.

## 6. Non-Functional Requirements

**Security (minimal but sane, not enterprise-grade)**
- Passwords hashed with bcrypt (or argon2id); never logged or stored in plaintext; minimum length ~8 characters.
- Session cookies: `HttpOnly`, `SameSite=Lax`, `Secure` in production; expiry; invalidated on logout.
- Input validation at the API boundary (required fields, email format, length limits); parameterized queries (no SQL injection); output escaping (no stored XSS in notes/task text).
- All task/note/reminder routes require an authenticated session, enforced server-side (not just hidden in the UI).
- Out of scope: MFA, SSO/OAuth, rate-limiting infra, CAPTCHA, email-based password reset.

**Other NFRs**
- Performance: trivial/local-scale — interactions should feel responsive (<1s); no load targets.
- Platform: latest evergreen desktop/mobile browsers (Chrome/Edge/Firefox); no legacy browser or native app support.
- Persistence: durable across server restarts (SQLite file); no backup/replication requirements.
- Accessibility: basic only — semantic HTML, labeled form inputs, keyboard-operable forms/buttons.

## 7. Acceptance Criteria / Definition of Done

- [ ] A new user can sign up with email + password.
- [ ] A user can log in with correct credentials and cannot log in with incorrect ones.
- [ ] Passwords are stored hashed, never in plaintext.
- [ ] An unauthenticated user cannot view or modify any task/note/reminder data (server-side enforced).
- [ ] A user can create, edit, and delete a task, and toggle it complete/incomplete.
- [ ] A user can create, edit, and delete a note.
- [ ] A user can create a reminder with a date/time, and it is visually flagged as due/overdue once that time passes.
- [ ] A user can log out, and the session is invalidated (subsequent authenticated requests fail).
- [ ] Data persists across a server restart / page reload.
- [ ] Each user only sees their own tasks/notes/reminders (verified data isolation between two test accounts).

## 8. Open Questions for Implementation Phase

- Exact DB schema (tables/columns) for `users`, `tasks`, `notes`, `reminders` — left to the implementing agent(s) to define against this PRD.
- Whether reminders are a field on `tasks` only, or also a standalone entity — current recommendation: support both (task-attached and standalone) per §4.4.
