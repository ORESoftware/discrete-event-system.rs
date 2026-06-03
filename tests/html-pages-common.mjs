import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const outDir = path.join(repoRoot, 'out');
export const browserLaunchArgs = [
  '--allow-file-access-from-files',
  '--disable-dev-shm-usage',
  '--disable-setuid-sandbox',
  '--disable-web-security',
  '--ignore-certificate-errors',
  '--no-sandbox',
];

export function discoverHtmlPages(dir = outDir) {
  assert.ok(fs.existsSync(dir), `Missing generated site directory: ${dir}`);

  const pages = [];
  const visit = (currentDir) => {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const entryPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        visit(entryPath);
      } else if (entry.isFile() && entry.name.endsWith('.html')) {
        pages.push(entryPath);
      }
    }
  };

  visit(dir);
  pages.sort((left, right) => htmlRelativePath(left).localeCompare(htmlRelativePath(right)));
  assert.ok(pages.length > 0, `No HTML pages found under ${dir}`);
  return pages;
}

export function extraHttpHeaders() {
  return process.env.HTML_TEST_AUTH ? { Auth: process.env.HTML_TEST_AUTH } : {};
}

export function findBrowserExecutable() {
  const candidates = [
    process.env.HTML_TEST_BROWSER,
    process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    process.env.PUPPETEER_EXECUTABLE_PATH,
    ...playwrightCacheCandidates(),
    ...systemBrowserCandidates(),
  ].filter(Boolean);

  const executable = candidates.find((candidate) => {
    try {
      return fs.existsSync(candidate) && fs.statSync(candidate).isFile();
    } catch {
      return false;
    }
  });

  if (!executable) {
    throw new Error(
      [
        'No Chromium executable found for HTML page tests.',
        'Set HTML_TEST_BROWSER to a Chrome/Chromium executable, or install Playwright browsers.',
      ].join(' '),
    );
  }

  return executable;
}

export function formatResult(framework, result) {
  const player = result.hasPlayerEvidence ? 'player=yes' : 'player=no';
  const controls = `controls=${result.controls}`;
  const surfaces = `canvas=${result.canvas} svg=${result.svg}`;
  return `ok ${framework} ${result.relativePath} ${player} ${controls} ${surfaces}`;
}

export function htmlRelativePath(filePath) {
  return path.relative(outDir, filePath).split(path.sep).join('/');
}

export function pageTarget(filePath) {
  const baseUrl = process.env.HTML_TEST_BASE_URL;
  if (!baseUrl) {
    return pathToFileURL(filePath).href;
  }

  const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  return new URL(htmlRelativePath(filePath), normalizedBase).href;
}

export function requiresPlayerEvidence(filePath) {
  const relativePath = htmlRelativePath(filePath);
  if (relativePath === 'index.html' || relativePath.endsWith('/report.html')) {
    return false;
  }

  return /animation|player|mppt|traffic|elevator|delivery|dc-motor|temp-control|obs-ctrl|vehicle-jump|hybrid\/|bouncing-ball|closed-loop|soccer-|two-disease|numerical-solvers|calculus-of-variations|decision\/(exec|hybrid|mdp|pomdp|studio)|plugin\/(lp|queue)|factmachine/i.test(
    relativePath,
  );
}

export async function smokeCheckPage(page, filePath, framework) {
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error?.message ?? String(error)));

  const target = pageTarget(filePath);
  const response = await page.goto(target, {
    timeout: Number(process.env.HTML_TEST_TIMEOUT_MS ?? 120000),
    waitUntil: 'domcontentloaded',
  });
  const status = typeof response?.status === 'function' ? response.status() : null;
  await wait(Number(process.env.HTML_TEST_SETTLE_MS ?? 200));

  const result = await page.evaluate(() => {
    const textNodes = Array.from(
      document.querySelectorAll('h1,h2,h3,title,button,label,output,a,summary,figcaption'),
    )
      .slice(0, 250)
      .map((node) => node.textContent ?? '')
      .join(' ');
    const bodyText = document.body?.textContent?.trim() ?? '';
    const playerDataIds = ['anim-data', 'player-data', 'plugin-payload'];
    const dataScripts = document.querySelectorAll('script[type="application/json"],script[type="application/x-ndjson"]').length;
    const buttons = document.querySelectorAll('button').length;
    const ranges = document.querySelectorAll('input[type="range"]').length;
    const controls = document.querySelectorAll('button,input,select,textarea').length;
    const canvas = document.querySelectorAll('canvas').length;
    const svg = document.querySelectorAll('svg').length;
    const playerData = playerDataIds.filter((id) => document.getElementById(id)).length;
    const playbackText = /play|pause|scrub|timeline|frame|speed|mppt|simulation|animate|player/i.test(textNodes);
    const hasRenderableContent = Boolean(document.body) && (document.body.children.length > 0 || bodyText.length > 0);
    const hasPlayerEvidence = Boolean(playerData || dataScripts || ranges || buttons || canvas || svg || playbackText);

    return {
      buttons,
      canvas,
      controls,
      dataScripts,
      hasPlayerEvidence,
      hasRenderableContent,
      playerData,
      ranges,
      readyState: document.readyState,
      svg,
      title: document.title,
    };
  });

  const relativePath = htmlRelativePath(filePath);
  assert.ok(status === null || (status >= 200 && status < 400), `${framework} ${relativePath} returned HTTP ${status}`);
  assert.notEqual(result.readyState, 'loading', `${framework} ${relativePath} did not finish parsing`);
  assert.equal(result.hasRenderableContent, true, `${framework} ${relativePath} rendered an empty body`);
  assert.equal(pageErrors.length, 0, `${framework} ${relativePath} page errors: ${pageErrors.join('; ')}`);

  if (requiresPlayerEvidence(filePath)) {
    assert.equal(result.hasPlayerEvidence, true, `${framework} ${relativePath} is missing player or animation evidence`);
  }

  return {
    ...result,
    relativePath,
    target,
  };
}

function playwrightCacheCandidates() {
  const roots = [
    path.join(os.homedir(), 'Library', 'Caches', 'ms-playwright'),
    path.join(os.homedir(), '.cache', 'ms-playwright'),
  ];
  const candidates = [];

  for (const root of roots) {
    if (!fs.existsSync(root)) {
      continue;
    }

    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory() || !entry.name.startsWith('chromium')) {
        continue;
      }

      const browserDir = path.join(root, entry.name);
      candidates.push(
        path.join(browserDir, 'chrome-mac', 'headless_shell'),
        path.join(browserDir, 'chrome-mac', 'Chromium.app', 'Contents', 'MacOS', 'Chromium'),
        path.join(browserDir, 'chrome-linux', 'headless_shell'),
        path.join(browserDir, 'chrome-linux', 'chrome'),
        path.join(browserDir, 'chrome-win', 'headless_shell.exe'),
        path.join(browserDir, 'chrome-win', 'chrome.exe'),
      );
    }
  }

  return candidates.sort((left, right) => right.localeCompare(left, undefined, { numeric: true }));
}

function systemBrowserCandidates() {
  return [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ];
}

function wait(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}
