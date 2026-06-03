import puppeteer from 'puppeteer-core';
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
const browser = await puppeteer.launch({
  args: browserLaunchArgs,
  executablePath,
  headless: true,
});
const failures = [];

console.log(`puppeteer chromium=${executablePath}`);
console.log(`puppeteer pages=${pages.length}`);

try {
  for (const filePath of pages) {
    const page = await browser.newPage();
    await page.setExtraHTTPHeaders(extraHttpHeaders());
    await page.setViewport({ height: 800, width: 1280 });

    try {
      const result = await smokeCheckPage(page, filePath, 'puppeteer');
      console.log(formatResult('puppeteer', result));
    } catch (error) {
      failures.push(error);
      console.error(`not ok puppeteer ${filePath}`);
      console.error(error?.stack ?? error);
    } finally {
      await page.close();
    }
  }
} finally {
  await browser.close();
}

if (failures.length > 0) {
  throw new Error(`Puppeteer HTML smoke tests failed on ${failures.length} page(s).`);
}
