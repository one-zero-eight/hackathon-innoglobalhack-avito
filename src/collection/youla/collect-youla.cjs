const fs = require('node:fs');

const [output, ...cities] = process.argv.slice(2);
const minimumLength = Number(process.env.MIN_LENGTH ?? 750);
const target = Number(process.env.TARGET_ROWS ?? Number.MAX_SAFE_INTEGER);
const workers = 3;
const maxVisits = Number(process.env.MAX_VISITS ?? 18);
const maxScrolls = Number(process.env.MAX_SCROLLS ?? 1);
const cachePath = output + '.cache';
const readRows = path => path && fs.existsSync(path)
  ? fs.readFileSync(path, 'utf8').split('\n').filter(Boolean).map(JSON.parse)
  : [];

function cleanDescription(text) {
  return text.split(/(?:^|\n)\s*(?:отзывы(?: клиентов| о работе)?|(?:меня|нас) можно найти по запросам|поисковые запросы|ключевые слова)(?=\s*[:!—–-]|\s*(?:\n|$))/iu)[0]
    .replace(/(?:\+7|8)[\s(.-]*\d{3}[\s).-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}/g, '[phone]')
    .replace(/\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b/gi, '[email]').trim();
}

const normalize = text => text.toLowerCase().replace(/ё/g, 'е').replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
const listingId = url => new URL(url).pathname.match(/-([a-f0-9]{24})$/)?.[1];

const categories = [
  [101, 'Ремонт квартир'],
  [102, 'Штукатурные работы'],
  [103, 'Укладка плитки'],
  [104, 'Услуги сантехника'],
  [105, 'Услуги электрика'],
  [106, 'Поклейка обоев'],
  [106, 'Малярные работы'],
];

async function main() {
  if (!output || !cities.length || cities.some(city => !/^[a-z0-9_-]+$/.test(city))) {
    throw new Error('Usage: collect-youla.cjs OUTPUT CITY [CITY ...]');
  }
  if (![minimumLength, target, maxVisits, maxScrolls].every(n => Number.isSafeInteger(n) && n > 0)) {
    throw new Error('MIN_LENGTH, TARGET_ROWS, MAX_VISITS and MAX_SCROLLS must be positive integers');
  }
  const idsByType = new Map(categories.map(([id, type]) => [type, id]));
  const existing = readRows(output);
  const saved = new Set(existing.map(row => listingId(row.url)));
  const descriptions = new Set(existing.map(row => normalize(row.description)));
  const seen = new Set(saved);
  let total = existing.length;
  let fetched = 0;
  let stop = false;
  let accessChallenge = false;
  let interrupted = false;
  for (const signal of ['SIGTERM', 'SIGINT']) {
    process.once(signal, () => {
      interrupted = true;
      stop = true;
    });
  }

  function accept(row) {
    const id = listingId(row.url);
    const key = normalize(row.description);
    if (total >= target || !id || saved.has(id) || descriptions.has(key) || [...row.description].length < minimumLength) return;
    fs.appendFileSync(output, JSON.stringify(row) + '\n');
    saved.add(id);
    descriptions.add(key);
    total++;
  }

  function record(fields, url, cache) {
    const id = listingId(url);
    const sampleMcId = idsByType.get(fields.source_type);
    if (!id || !fields.description) return;
    const sourceDescription = fields.description
      .replace(/(?:\+7|8)[\s(.-]*\d{3}[\s).-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}/g, '[phone]')
      .replace(/\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b/gi, '[email]');
    if (cache) fs.appendFileSync(cachePath, JSON.stringify({
      ...fields, url: url.split('?')[0], description: sourceDescription,
    }) + '\n');
    seen.add(id);
    const description = cleanDescription(sourceDescription);
    if (!sampleMcId || !fields.title || !description || /сделка состоялась|[⭐★]{3,}/iu.test(description)) return;
    const title = fields.title.replace(/\s+[–—-]\s+.*(?:цена|дата размещения|истекает).*$/iu, '').trim();
    const row = { platform: 'youla', url: url.split('?')[0], sample_mc_id: sampleMcId, title, description, source_type: fields.source_type };
    accept(row);
  }

  for (const row of readRows(cachePath)) record(row, row.url, false);
  for (const path of (process.env.SEED_FILES ?? '').split(',')) {
    for (const row of readRows(path)) {
      if (!seen.has(listingId(row.url))) record(row, row.url, true);
    }
  }
  console.error(`Starting with ${total} qualifying rows`);
  if (total >= target) return;

  const browser = await require('playwright').chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1440, height: 1600 } });
  await context.route('**/*', route => {
    const request = route.request();
    if (['image', 'font', 'media'].includes(request.resourceType())) return route.abort();
    return route.continue();
  });

  async function navigate(page, url) {
    const response = await page.goto(url, { waitUntil: 'commit', timeout: 10000 });
    if ([403, 429].includes(response?.status()) || /captcha|verify/i.test(new URL(page.url()).pathname)) {
      accessChallenge = true;
      stop = true;
      throw new Error('Access challenge; stopping collection');
    }
    return response?.ok();
  }

  const jobs = cities.flatMap(city => categories.map(([category, filter]) => ({ city, category, filter })));
  async function worker() {
    const catalog = await context.newPage();
    const item = await context.newPage();
    while (!stop && total < target && jobs.length) {
      const { city, category, filter } = jobs.shift();
      const catalogUrl = `https://youla.ru/${city}/uslugi/remont-stroitelstvo`;
      try {
        if (!await navigate(catalog, catalogUrl)) continue;
        const label = catalog.locator('label').filter({ hasText: new RegExp('^' + filter + '$') });
        await label.evaluate(node => node.click(), { timeout: 5000 });
        await label.locator('[data-test-value="checked"]').waitFor({ timeout: 5000 });
        const cardSelector = 'a[href*="source_view=catalog"]';
        const cards = catalog.locator(cardSelector);
        await cards.first().waitFor({ state: 'attached', timeout: 5000 });
        const discovered = new Map();
        async function collectLinks() {
          const links = await cards.evaluateAll(nodes => nodes.map(node => node.href));
          for (const url of links) {
            if (new URL(url).origin !== 'https://youla.ru') continue;
            const id = listingId(url);
            if (id) discovered.set(id, url);
          }
        }
        await collectLinks();
        console.error(city, filter, `initial catalog: ${discovered.size} IDs`);
        let emptyScrolls = 0;
        for (let scroll = 1; scroll <= maxScrolls && !stop; scroll++) {
          const before = discovered.size;
          try {
            await cards.last().scrollIntoViewIfNeeded({ timeout: 2000 });
            const growth = await catalog.waitForFunction(({ selector, knownIds }) => {
              const known = new Set(knownIds);
              return Array.from(document.querySelectorAll(selector)).some(link => {
                const id = link.pathname.match(/-([a-f0-9]{24})$/)?.[1];
                return link.origin === location.origin && id && !known.has(id);
              });
            }, { selector: cardSelector, knownIds: [...discovered.keys()] },
            { polling: 200, timeout: 3500 });
            await growth.dispose();
          } catch (error) {
            if (error.name !== 'TimeoutError' && !error.message.includes('not attached to the DOM')) throw error;
          }
          await collectLinks();
          const added = discovered.size - before;
          const unseen = [...discovered.keys()].filter(id => !seen.has(id)).length;
          console.error(city, filter, `scroll ${scroll}/${maxScrolls}: +${added} IDs, ${discovered.size} total, ${unseen} unseen`);
          emptyScrolls = added ? 0 : emptyScrolls + 1;
          if (emptyScrolls === 2) break;
          if (!added && !stop && scroll < maxScrolls) await catalog.mouse.wheel(0, 1200);
        }
        const links = [...discovered.values()];
        links.sort((a, b) => Number(new URL(b).pathname.startsWith('/' + city + '/')) - Number(new URL(a).pathname.startsWith('/' + city + '/')));
        let attempts = 0;
        let accepted = 0;
        for (const url of links) {
          if (stop || total >= target || attempts >= maxVisits) break;
          if (new URL(url).origin !== 'https://youla.ru') continue;
          const id = listingId(url);
          if (!id || seen.has(id)) continue;
          seen.add(id);
          attempts++;
          fetched++;
          try {
            if (!await navigate(item, url)) continue;
            const result = await item.waitForFunction(() => {
              const labels = [...document.querySelectorAll('dt')];
              const value = name => labels.find(node => node.textContent.trim() === name)?.nextElementSibling?.innerText.trim();
              const description = value('Описание');
              const source_type = value('Вид услуг');
              if (!description) return false;
              const title = document.querySelector('h1')?.textContent.trim() || document.title.split(' – ')[0];
              return { title, description, source_type };
            }, undefined, { polling: 200, timeout: 10000 });
            const fields = await result.jsonValue();
            await result.dispose();
            const before = total;
            if (!stop) record(fields, item.url(), true);
            accepted += total - before;
          } catch (error) {
            if (interrupted) break;
            if (!browser.isConnected()) {
              stop = true;
              throw error;
            }
            console.error(id, error.message.split('\n')[0]);
          }
        }
        console.error(city, category, `${attempts} detail visits, ${accepted} accepted, ${total} total`);
      } catch (error) {
        if (!interrupted) console.error(city, filter, error.message.split('\n')[0]);
        if (!interrupted && !browser.isConnected()) {
          stop = true;
          throw error;
        }
      }
    }
  }

  try {
    await Promise.all(Array.from({ length: workers }, worker));
  } finally {
    await browser.close();
  }
  console.error(`Saved ${total} total rows after ${fetched} detail visits`);
  if (interrupted) process.exitCode = 130;
  else if (accessChallenge) process.exitCode = 1;
}
if (process.argv[2] === '--self-check') {
  const assert = require('node:assert/strict');
  assert.equal(cleanDescription('Работы по ремонту.\n\nОТЗЫВЫ\n5.0 Сделка состоялась'), 'Работы по ремонту.');
  assert.equal(cleanDescription('Гарантия.\nМеня можно найти по запросам:\nэлектрик, электрик'), 'Гарантия.');
  assert.equal(cleanDescription('О нашей работе есть отзывы.\nЗаменяем проводку.'), 'О нашей работе есть отзывы.\nЗаменяем проводку.');
} else main().catch(error => { console.error(error.message); process.exitCode = 1; });
