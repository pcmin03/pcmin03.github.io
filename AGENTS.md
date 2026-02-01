# Repository Guidelines

## Project Structure & Module Organization
- Jekyll theme sources live in `_layouts/`, `_includes/`, `_sass/`, and `_data/`.
- Site content examples are in `_posts/`, `pages/`, and top-level `.html`/`.md` pages (e.g., `index.html`).
- Documentation site sources live under `docs/` (separate config in `docs/_config.yml`).
- A demo/test site exists in `test/` with its own `_config.yml` and content.
- Build output typically goes to `_site/` (Jekyll default).
- Assets such as images, fonts, and compiled CSS/JS live in `assets/`.

## Build, Test, and Development Commands
- `bundle install` installs Ruby/Jekyll dependencies from `Gemfile`.
- `bundle exec jekyll serve -H 0.0.0.0` runs the theme locally.
- `npm run dev` serves the docs site using `docs/_config.dev.yml`.
- `npm run demo-dev` serves the docs/demo site using `docs/_config.yml`.
- `npm run build` builds the production site with `JEKYLL_ENV=production`.
- `npm run eslint` and `npm run stylelint` run JS and SCSS linting.
- Docker workflows are available via `npm run docker-dev:default` and related scripts in `package.json`.

## Coding Style & Naming Conventions
- JavaScript linting is enforced by ESLint (`.eslintrc`): 2-space indent, single quotes, semicolons, no `console`, camelCase, and trailing commas disallowed.
- SCSS linting is enforced by Stylelint (`.stylelintrc`) with ordered properties and double quotes.
- Jekyll naming follows standard conventions: layouts in `_layouts/`, includes in `_includes/`, posts as `_posts/YYYY-MM-DD-title.md`.

## Testing Guidelines
- There is no automated test framework configured; rely on local Jekyll builds and linting.
- Use the demo site in `test/` or `docs/` to validate layout changes.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits via commitlint (`.commitlintrc.js`).
  Example: `feat(layout): add hero banner`.
- Valid types include `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `build`, and `release`.
- Keep headers <= 72 chars and use lowercase type/scope.
- PRs should include a brief description, linked issues when applicable, and screenshots for visual changes.

## Configuration Tips
- Use `JEKYLL_ENV=production|beta` for environment-specific behavior.
- When editing docs content, update both `docs/` content and configs as needed.
