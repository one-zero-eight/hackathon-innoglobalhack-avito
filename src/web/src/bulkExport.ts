import type { ImportedAd } from './adsParser'
import type { BulkRow } from './bulkTypes'

const CSV_HEADER = [
  'index',
  'description',
  'source_mcId',
  'target_mc_ids',
  'draft_title',
  'draft_text',
] as const

/** Экранирование ячейки CSV по RFC 4180 (кавычки, переводы строк, разделитель). */
function csvCell(value: string): string {
  return /["\n\r,]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value
}

/**
 * JSON-экспорт пакетных результатов: массив
 * `{index, description, mcId, drafts | error}` по строкам входного файла.
 */
export function buildBulkJson(ads: ImportedAd[], rows: BulkRow[]): string {
  const items = ads.map((ad, index) => {
    const base = { index, description: ad.description, mcId: ad.mcId }
    const row = rows[index]
    if (row === undefined || row.error !== null) {
      return { ...base, error: row?.error ?? 'Обработка не выполнялась.' }
    }
    return { ...base, drafts: row.drafts }
  })

  return JSON.stringify(items, null, 2)
}

/**
 * CSV-экспорт пакетных результатов: по строке на черновик; строки без
 * черновиков (ошибка или остановка) выводятся одной строкой с пустыми
 * черновиковыми полями. Колонки: index, description, source_mcId,
 * target_mc_ids, draft_title, draft_text.
 */
export function buildBulkCsv(ads: ImportedAd[], rows: BulkRow[]): string {
  const lines: string[] = [CSV_HEADER.join(',')]

  ads.forEach((ad, index) => {
    const row = rows[index]
    const targets = (row?.drafts ?? []).map((draft) => draft.mcId).join(';')
    const base = [String(index), ad.description, String(ad.mcId), targets]

    if (row === undefined || row.drafts.length === 0) {
      lines.push([...base, '', ''].map(csvCell).join(','))
      return
    }

    for (const draft of row.drafts) {
      lines.push([...base, draft.title ?? '', draft.text ?? ''].map(csvCell).join(','))
    }
  })

  // BOM: корректное отображение кириллицы в Excel.
  return `\uFEFF${lines.join('\r\n')}\r\n`
}

/** Скачивает текстовый файл через Blob + download, без зависимостей. */
export function downloadTextFile(fileName: string, mime: string, content: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mime }))
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/** Скачивает пакетные результаты в JSON. */
export function downloadBulkJson(ads: ImportedAd[], rows: BulkRow[]): void {
  downloadTextFile('bulk-drafts.json', 'application/json;charset=utf-8', buildBulkJson(ads, rows))
}

/** Скачивает пакетные результаты в CSV. */
export function downloadBulkCsv(ads: ImportedAd[], rows: BulkRow[]): void {
  downloadTextFile('bulk-drafts.csv', 'text/csv;charset=utf-8', buildBulkCsv(ads, rows))
}
