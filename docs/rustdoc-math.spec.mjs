import { test, expect, devices } from "@playwright/test";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function loadTargets() {
  const p = path.join(__dirname, "rustdoc-math-targets.json");
  const raw = await readFile(p, "utf8");
  return JSON.parse(raw).targets;
}

function fileUrl(p) {
  // Ensure proper file:// URL formatting across platforms.
  const normalized = path.resolve(p).replace(/\\/g, "/");
  return `file://${normalized}`;
}

async function countLikelyMathDelimiters(page) {
  // Heuristic: count occurrences of LaTeX delimiters in *text nodes* outside code/pre.
  // We only use this to decide whether to require the presence of rendered math nodes.
  return await page.evaluate(() => {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let count = 0;
    let node;
    while ((node = walker.nextNode())) {
      const parent = node.parentElement;
      if (!parent) continue;
      if (parent.closest("code, pre, script, style")) continue;

      const t = node.nodeValue || "";
      // Cheap signals: $...$, $$...$$, \( \), \[ \]
      // (We do not attempt to parse; we just want "something that looks like math exists".)
      if (t.includes("\\(") || t.includes("\\)") || t.includes("\\[") || t.includes("\\]")) count++;
      if (t.includes("$$")) count++;
      // Count single-$ only if it appears more than once (avoid currency).
      const dollars = (t.match(/\$/g) || []).length;
      if (dollars >= 2) count++;
    }
    return count;
  });
}

async function assertNoRendererErrors(page) {
  // KaTeX: errors become `.katex-error` elements.
  await expect(page.locator(".katex-error")).toHaveCount(0);

  // MathJax: no universal "error node", so at minimum ensure it produced containers
  // when we think math exists (checked separately), and avoid the most common visible marker.
  await expect(page.locator("text=/MathJax.*error/i")).toHaveCount(0);
}

async function assertMathRenderedIfPresent(page) {
  const likely = await countLikelyMathDelimiters(page);
  if (likely === 0) return;

  // Give renderers a moment (CDN + DOM mutation observers).
  await page.waitForTimeout(250);

  const katexCount = await page.locator(".katex").count();
  const mjxCount = await page.locator("mjx-container").count();

  expect(
    katexCount + mjxCount,
    "Math delimiters detected, but neither KaTeX (.katex) nor MathJax (mjx-container) produced output.",
  ).toBeGreaterThan(0);
}

async function checkPage(page, url, screenshotName, testInfo) {
  await page.goto(url, { waitUntil: "load" });
  await assertNoRendererErrors(page);
  await assertMathRenderedIfPresent(page);
  // We intentionally do NOT use Playwright snapshot assertions yet.
  // At this stage, screenshots are an audit artifact, not a golden test.
  await page.screenshot({
    path: testInfo.outputPath(screenshotName),
    fullPage: true,
  });
}

test.describe("rustdoc math rendering (batch)", () => {
  test("desktop chromium (file://)", async ({ page }, testInfo) => {
    const targets = await loadTargets();
    for (const t of targets) {
      const indexPath = path.join(t.docRoot, t.crate, "index.html");
      const allPath = path.join(t.docRoot, t.crate, "all.html");

      await checkPage(page, fileUrl(indexPath), `${t.id}__index__desktop.png`, testInfo);
      await checkPage(page, fileUrl(allPath), `${t.id}__all__desktop.png`, testInfo);
    }
  });

  test("mobile emulation (iPhone 14, file://)", async ({ browser }, testInfo) => {
    const targets = await loadTargets();
    const ctx = await browser.newContext(devices["iPhone 14"]);
    const page = await ctx.newPage();

    for (const t of targets) {
      const indexPath = path.join(t.docRoot, t.crate, "index.html");
      const allPath = path.join(t.docRoot, t.crate, "all.html");

      await checkPage(page, fileUrl(indexPath), `${t.id}__index__iphone14.png`, testInfo);
      await checkPage(page, fileUrl(allPath), `${t.id}__all__iphone14.png`, testInfo);
    }
  });
});

