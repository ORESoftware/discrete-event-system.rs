import { chromium } from 'playwright-core';
import {
  browserLaunchArgs,
  discoverHtmlPages,
  extraHttpHeaders,
  findBrowserExecutable,
  formatResult,
  smokeCheckPage,
} from './html-pages-common.mjs';

const pages = discoverHtmlPages();
const executablePath = findBrowserExecutable();
const browser = await chromium.launch({
  args: browserLaunchArgs,
  executablePath,
  headless: true,
});
const failures = [];

console.log(`playwright chromium=${executablePath}`);
console.log(`playwright pages=${pages.length}`);

try {
  for (const filePath of pages) {
    const context = await browser.newContext({
      extraHTTPHeaders: extraHttpHeaders(),
      ignoreHTTPSErrors: true,
      viewport: { height: 800, width: 1280 },
    });
    const page = await context.newPage();

    try {
      const result = await smokeCheckPage(page, filePath, 'playwright');
      console.log(formatResult('playwright', result));
    } catch (error) {
      failures.push(error);
      console.error(`not ok playwright ${filePath}`);
      console.error(error?.stack ?? error);
    } finally {
      await context.close();
    }
  }
} finally {
  await browser.close();
}

if (failures.length > 0) {
  throw new Error(`Playwright HTML smoke tests failed on ${failures.length} page(s).`);
}
