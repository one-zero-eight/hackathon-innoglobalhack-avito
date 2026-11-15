export interface ImportedAd {
  description: string
  mcId: number
  title?: string
}

interface CsvRow {
  cells: string[]
  line: number
}

function parseAd(value: unknown, rowLabel: string): ImportedAd {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${rowLabel}: ожидался объект объявления.`)
  }

  const record = value as Record<string, unknown>
  if (typeof record.description !== 'string' || record.description.trim() === '') {
    throw new Error(`${rowLabel}: поле description должно быть непустой строкой.`)
  }

  const rawMcId = record.mcId
  const mcId =
    typeof rawMcId === 'number'
      ? rawMcId
      : typeof rawMcId === 'string' && rawMcId.trim() !== ''
        ? Number(rawMcId)
        : Number.NaN
  if (!Number.isInteger(mcId) || mcId <= 0) {
    throw new Error(`${rowLabel}: поле mcId должно быть положительным целым числом.`)
  }

  if (record.title !== undefined && typeof record.title !== 'string') {
    throw new Error(`${rowLabel}: поле title должно быть строкой.`)
  }

  return {
    description: record.description.trim(),
    mcId,
    ...(typeof record.title === 'string' ? { title: record.title.trim() } : {}),
  }
}

function detectCsvDelimiter(text: string): ',' | ';' {
  let commas = 0
  let semicolons = 0
  let quoted = false

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]
    if (char === '"') {
      if (quoted && text[index + 1] === '"') {
        index += 1
      } else {
        quoted = !quoted
      }
    } else if (!quoted && (char === '\n' || char === '\r')) {
      break
    } else if (!quoted && char === ',') {
      commas += 1
    } else if (!quoted && char === ';') {
      semicolons += 1
    }
  }

  return semicolons > commas ? ';' : ','
}

function parseCsvRows(text: string, delimiter: ',' | ';'): CsvRow[] {
  const rows: CsvRow[] = []
  let cells: string[] = []
  let field = ''
  let quoted = false
  let line = 1
  let rowLine = 1

  const finishRow = () => {
    cells.push(field)
    if (cells.some((cell) => cell.trim() !== '')) {
      rows.push({ cells, line: rowLine })
    }
    cells = []
    field = ''
  }

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]

    if (char === '"') {
      if (quoted && text[index + 1] === '"') {
        field += '"'
        index += 1
      } else {
        quoted = !quoted
      }
    } else if (char === delimiter && !quoted) {
      cells.push(field)
      field = ''
    } else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && text[index + 1] === '\n') {
        index += 1
      }
      finishRow()
      line += 1
      rowLine = line
    } else if ((char === '\n' || char === '\r') && quoted) {
      if (char === '\r' && text[index + 1] === '\n') {
        index += 1
      }
      field += '\n'
      line += 1
    } else {
      field += char
    }
  }

  if (quoted) {
    throw new Error(`Строка ${rowLine}: не закрыта двойная кавычка в CSV.`)
  }
  finishRow()
  return rows
}

function parseCsv(text: string): ImportedAd[] {
  const rows = parseCsvRows(text, detectCsvDelimiter(text))
  if (rows.length === 0) {
    return []
  }

  const headers = rows[0].cells.map((cell) => cell.trim().toLocaleLowerCase('en'))
  const descriptionIndex = headers.indexOf('description')
  const mcIdIndex = headers.indexOf('mcid')
  const titleIndex = headers.indexOf('title')
  if (descriptionIndex === -1 || mcIdIndex === -1) {
    throw new Error('В CSV должны быть колонки description и mcId; колонка title необязательна.')
  }

  return rows.slice(1).map(({ cells, line }) => {
    const value: Record<string, unknown> = {
      description: cells[descriptionIndex]?.trim() ?? '',
      mcId: cells[mcIdIndex]?.trim() ?? '',
    }
    if (titleIndex !== -1) {
      value.title = cells[titleIndex]?.trim() ?? ''
    }
    return parseAd(value, `Строка ${line}`)
  })
}

function parseJson(text: string): ImportedAd[] {
  if (text.startsWith('[')) {
    let value: unknown
    try {
      value = JSON.parse(text)
    } catch {
      throw new Error('Не удалось разобрать JSON: проверьте синтаксис массива.')
    }
    if (!Array.isArray(value)) {
      throw new Error('JSON должен содержать массив объявлений.')
    }
    return value.map((item, index) => parseAd(item, `Объявление ${index + 1}`))
  }

  const ads: ImportedAd[] = []
  for (const [index, line] of text.split('\n').entries()) {
    const trimmed = line.trim()
    if (trimmed === '') {
      continue
    }
    let value: unknown
    try {
      value = JSON.parse(trimmed)
    } catch {
      throw new Error(`Строка ${index + 1}: не удалось разобрать JSON-объект.`)
    }
    ads.push(parseAd(value, `Строка ${index + 1}`))
  }
  return ads
}

/** Разобрать CSV, JSON-массив или JSONL в объявления для подстановки в форму. */
export function parseAdsFile(contents: string, fileName: string): ImportedAd[] {
  const text = contents.replace(/^\uFEFF/, '').trim()
  if (text === '') {
    throw new Error('Файл пуст. Добавьте хотя бы одно объявление.')
  }

  const extension = fileName.split('.').pop()?.toLocaleLowerCase('en')
  let ads: ImportedAd[]
  if (extension === 'csv') {
    ads = parseCsv(text)
  } else if (extension === 'json' || extension === 'jsonl') {
    ads = parseJson(text)
  } else {
    throw new Error('Неподдерживаемый формат файла. Выберите CSV, JSON или JSONL.')
  }

  if (ads.length === 0) {
    throw new Error('В файле нет объявлений.')
  }
  return ads
}
