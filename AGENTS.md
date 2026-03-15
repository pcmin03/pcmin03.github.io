# Repository Guidelines

## Project Structure & Module Organization
This repository is a Jekyll theme with a local site layered on top. Theme templates live in `_layouts/` and reusable Liquid partials in `_includes/`. Styles are organized under `_sass/`, with shared components in `_sass/common/` and skin variants in `_sass/skins/`. Client-side scripts live in `_includes/scripts/`. Root-level content such as `_posts/`, `assets/`, and `_config.yml` powers the local site, while `docs/` contains the theme demo/documentation site and `test/` contains a fixture site used to exercise the gem-based theme.

## Build, Test, and Development Commands
Install dependencies with `bundle install` and `npm install`.

- `npm run dev`: serve the documentation/demo site with `docs/_config.dev.yml`.
- `npm run default`: serve the root site with live rebuilds.
- `npm run build`: production Jekyll build.
- `npm run eslint`: lint JavaScript in `_includes/**/*.js`.
- `npm run stylelint`: lint SCSS in `_sass/**/*.scss`.
- `npm run eslint-fix` / `npm run stylelint-fix`: apply safe autofixes.

Use `npm run demo-dev` when you need the demo configuration from `docs/_config.yml`.

## Coding Style & Naming Conventions
Follow `.editorconfig`: UTF-8, LF endings, final newline, and 2-space indentation. JavaScript follows `.eslintrc`: single quotes, required semicolons, camelCase identifiers, and 2-space indents. SCSS is checked by Stylelint; keep declarations ordered consistently and prefer lowercase short hex colors. Match existing naming patterns such as `_sass/common/components/_button.scss`, `_includes/scripts/components/search.js`, and BEM-like classes like `.button--primary`.

## Testing Guidelines
There is no formal unit-test suite in this repository. Validation is done through linting and manual preview checks.

- Run `npm run eslint` and `npm run stylelint` before opening a PR.
- Preview affected pages with `npm run dev` or `npm run default`.
- For theme-level changes, also verify the gem fixture under `test/` still renders correctly.

## Commit & Pull Request Guidelines
Commits are enforced by Husky and Commitlint, so prefer Conventional Commits such as `fix(nav): prevent horizontal shift` or `docs(readme): clarify demo setup`. Keep subjects imperative and under 72 characters.

PRs should include a short summary, linked issue when applicable, and screenshots or GIFs for layout, styling, or interaction changes. Note any config updates and list the commands you ran to validate the change.
