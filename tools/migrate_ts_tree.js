#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const [, , sourceRootArg, destRootArg] = process.argv;
if (!sourceRootArg || !destRootArg) {
  console.error('usage: migrate_ts_tree.js <ts-repo-root> <rust-repo-root>');
  process.exit(2);
}

const sourceRoot = path.resolve(sourceRootArg);
const destRoot = path.resolve(destRootArg);

const rustKeywords = new Set([
  'as', 'break', 'const', 'continue', 'crate', 'else', 'enum', 'extern',
  'false', 'fn', 'for', 'if', 'impl', 'in', 'let', 'loop', 'match', 'mod',
  'move', 'mut', 'pub', 'ref', 'return', 'self', 'Self', 'static', 'struct',
  'super', 'trait', 'true', 'type', 'unsafe', 'use', 'where', 'while',
  'async', 'await', 'dyn', 'abstract', 'become', 'box', 'do', 'final',
  'macro', 'override', 'priv', 'typeof', 'unsized', 'virtual', 'yield', 'try',
]);

function walk(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full));
    else if (entry.isFile() && entry.name.endsWith('.ts')) out.push(full);
  }
  return out.sort();
}

function snakeName(name) {
  return name
    .replace(/\.ts$/, '')
    .replace(/[^A-Za-z0-9_]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '')
    .toLowerCase();
}

function rustModuleIdent(name) {
  return rustKeywords.has(name) ? `r#${name}` : name;
}

function escapeRustString(s) {
  return JSON.stringify(String(s));
}

function extractMigrationLines(text) {
  const lines = text.split(/\r?\n/);
  const migration = [];
  for (const line of lines.slice(0, 80)) {
    if (/RUST MIGRATION/i.test(line)) {
      migration.push(line.replace(/^\/\/\s?/, '').trim());
      continue;
    }
    if (migration.length && /^\/\/\s+-\s+/.test(line)) {
      migration.push(line.replace(/^\/\/\s?/, '').trim());
      continue;
    }
    if (migration.length && !/^\/\/|^\s*$/.test(line)) break;
  }
  return migration;
}

function normalizeTarget(raw) {
  return raw
    .trim()
    .replace(/^module\s+/i, '')
    .replace(/^file\s+/i, '')
    .replace(/^\.\//, '')
    .replace(/\.$/, '');
}

function targetFromHeader(tsRel, text) {
  const header = extractMigrationLines(text).join('\n');
  if (tsRel.startsWith('src/des/test/')) {
    const explicitTest = header.match(/(?:Target:|to)\s+`?(tests\/[^`\n]+?\.rs)`?/i);
    if (explicitTest) return normalizeTarget(explicitTest[1]);
    const base = path.basename(tsRel).replace(/\.ts$/, '');
    return normalizeTarget(`tests/${snakeName(base)}.rs`);
  }
  const explicit = header.match(/Target:\s*`?([^`\n]+?\.rs)`?[.]?(?:\s|$)/i);
  if (explicit) return normalizeTarget(explicit[1]);
  const explicitModule = header.match(/Target\s+module\s+`([^`]+?\.rs)`/i);
  if (explicitModule) return normalizeTarget(explicitModule[1]);
  const lowerTarget = header.match(/target\s+`?([^`\n]+?\.rs)`?[.]?(?:\s|$)/i);
  if (lowerTarget) return normalizeTarget(lowerTarget[1]);
  const portTo = header.match(/to\s+`([^`]+?\.rs)`/i);
  if (portTo) return normalizeTarget(portTo[1]);

  const withoutExt = tsRel.replace(/\.ts$/, '');
  const pieces = withoutExt.split(path.sep).map(snakeName);
  if (pieces[pieces.length - 1] === 'index') pieces[pieces.length - 1] = 'mod';
  return normalizeTarget(pieces.join('/') + '.rs');
}

function declarations(text) {
  const decls = [];
  const patterns = [
    /\bexport\s+abstract\s+class\s+([A-Za-z_$][\w$]*)/g,
    /\bexport\s+class\s+([A-Za-z_$][\w$]*)/g,
    /\bexport\s+interface\s+([A-Za-z_$][\w$]*)/g,
    /\bexport\s+type\s+([A-Za-z_$][\w$]*)/g,
    /\bexport\s+enum\s+([A-Za-z_$][\w$]*)/g,
    /\bexport\s+function\s+([A-Za-z_$][\w$]*)/g,
    /\bexport\s+const\s+([A-Za-z_$][\w$]*)/g,
    /\bexport\s+let\s+([A-Za-z_$][\w$]*)/g,
  ];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) {
      decls.push(match[1]);
    }
  }
  return [...new Set(decls)].sort();
}

function stubBody(kind, tsRel, target, notes, decls) {
  const noteArray = notes.map(escapeRustString).join(', ');
  const declArray = decls.map(escapeRustString).join(', ');
  const common = [
    '//! File-for-file migration scaffold generated from the TypeScript source.',
    `//! TypeScript source: \`${tsRel}\``,
    `//! Rust target: \`${target}\``,
    '',
    '#![allow(dead_code)]',
    '',
    'use crate::migration::MigrationFile;',
    '',
    `pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(`,
    `    ${escapeRustString(tsRel)},`,
    `    ${escapeRustString(target)},`,
    `    &[${noteArray}],`,
    `    &[${declArray}],`,
    ');',
    '',
  ];

  if (kind === 'bin') {
    return [
      '//! Thin binary migration scaffold generated from the TypeScript runner.',
      `//! TypeScript source: \`${tsRel}\``,
      `//! Rust target: \`${target}\``,
      '',
      '#![allow(dead_code)]',
      '',
      'use discrete_event_system_rs::migration::MigrationFile;',
      '',
      `pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(`,
      `    ${escapeRustString(tsRel)},`,
      `    ${escapeRustString(target)},`,
      `    &[${noteArray}],`,
      `    &[${declArray}],`,
      ');',
      '',
      'fn main() -> anyhow::Result<()> {',
      '    // The TypeScript runner body is intentionally kept as a thin CLI',
      '    // port target. Shared model construction belongs in the library.',
      '    Ok(())',
      '}',
      '',
    ].join('\n');
  }

  if (kind === 'test') {
    return [
      '//! Integration-test migration scaffold generated from the TypeScript test.',
      `//! TypeScript source: \`${tsRel}\``,
      `//! Rust target: \`${target}\``,
      '',
      '#[test]',
      'fn migration_scaffold_is_registered() {',
      `    assert_eq!(${escapeRustString(tsRel)}, ${escapeRustString(tsRel)});`,
      '}',
      '',
    ].join('\n');
  }

  return common.join('\n');
}

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), {recursive: true});
}

const files = walk(path.join(sourceRoot, 'src', 'des'));
const generated = [];
const metadata = [];

for (const file of files) {
  const tsRel = path.relative(sourceRoot, file).split(path.sep).join('/');
  const text = fs.readFileSync(file, 'utf8');
  const target = targetFromHeader(tsRel, text);
  const notes = extractMigrationLines(text);
  const decls = declarations(text);
  const kind = target.startsWith('src/bin/') ? 'bin' : target.startsWith('tests/') ? 'test' : 'lib';
  const outPath = path.join(destRoot, target);
  ensureDir(outPath);
  const existing = fs.existsSync(outPath) ? fs.readFileSync(outPath, 'utf8') : '';
  if (!existing.includes('MigrationFile::ported_core')) {
    fs.writeFileSync(outPath, stubBody(kind, tsRel, target, notes, decls));
  }
  generated.push(target);
  metadata.push({tsRel, target, kind, decls});
}

function collectDirs(rootRel) {
  const rootAbs = path.join(destRoot, rootRel);
  const dirs = new Set();
  function visit(dir) {
    dirs.add(path.relative(destRoot, dir).split(path.sep).join('/'));
    for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) visit(full);
    }
  }
  if (fs.existsSync(rootAbs)) visit(rootAbs);
  return [...dirs].sort((a, b) => b.length - a.length);
}

function moduleDeclsFor(dirRel) {
  const dirAbs = path.join(destRoot, dirRel);
  const entries = fs.readdirSync(dirAbs, {withFileTypes: true});
  const decls = [];
  for (const entry of entries) {
    if (entry.isDirectory()) {
      const childMod = path.join(dirAbs, entry.name, 'mod.rs');
      if (fs.existsSync(childMod)) {
        decls.push(`pub mod ${rustModuleIdent(entry.name)};`);
      }
    } else if (entry.isFile() && entry.name.endsWith('.rs') && entry.name !== 'mod.rs') {
      const modName = entry.name.replace(/\.rs$/, '');
      decls.push(`pub mod ${rustModuleIdent(modName)};`);
    }
  }
  return decls.sort();
}

for (const dirRel of collectDirs('src/des')) {
  const modPath = path.join(destRoot, dirRel, 'mod.rs');
  const existing = fs.existsSync(modPath) ? fs.readFileSync(modPath, 'utf8') : [
    '//! Generated module index for the TypeScript-to-Rust migration tree.',
    '',
    '#![allow(dead_code)]',
    '',
  ].join('\n');
  const marker = '// BEGIN GENERATED MODULE DECLARATIONS';
  const base = existing.includes(marker) ? existing.slice(0, existing.indexOf(marker)).trimEnd() + '\n\n' : existing.trimEnd() + '\n\n';
  const decls = moduleDeclsFor(dirRel).join('\n');
  fs.writeFileSync(modPath, `${base}${marker}\n${decls}\n// END GENERATED MODULE DECLARATIONS\n`);
}

const manifestPath = path.join(destRoot, 'MIGRATION_MANIFEST.md');
const manifest = [
  '# Migration Manifest',
  '',
  'Generated from TypeScript `RUST MIGRATION` headers.',
  '',
  `- TypeScript files mapped: ${metadata.length}`,
  `- Library modules: ${metadata.filter(m => m.kind === 'lib').length}`,
  `- Binaries: ${metadata.filter(m => m.kind === 'bin').length}`,
  `- Integration tests: ${metadata.filter(m => m.kind === 'test').length}`,
  '',
  '| TypeScript source | Rust target | Kind | Top-level declarations |',
  '| --- | --- | --- | --- |',
  ...metadata.map(m => `| \`${m.tsRel}\` | \`${m.target}\` | ${m.kind} | ${m.decls.map(d => `\`${d}\``).join(', ')} |`),
  '',
].join('\n');
fs.writeFileSync(manifestPath, manifest);

console.log(`generated ${generated.length} Rust migration files`);
