/** Микрокатегория из GET /api/microcategories. */
export interface Microcategory {
  mcId: number
  title: string
}

/** Черновик объявления из POST /api/drafts. */
export interface Draft {
  mcId: number
  title: string | null
  text: string | null
}

/** Ответ POST /api/drafts. */
export interface DraftsResponse {
  drafts: Draft[]
}

/** Тело запроса POST /api/drafts. */
export interface DraftsRequest {
  description: string
  mcId: number
}

/** Сплит датасета кейса (GET /api/dataset/*). */
export type DatasetSplit = 'train' | 'test'

/** Объявление из датасета кейса (GET /api/dataset/random). */
export interface DatasetSample {
  id: number
  split: DatasetSplit
  itemId: number | null
  title: string
  description: string
  sourceMcId: number
  foundMcIds: number[] | null
  categories: string[] | null
}

/** Ошибка обращения к бэкенду с читаемым русским сообщением. */
export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status = 0) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function parseBody(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return null
  }
}

/**
 * Извлекает detail из ошибки бэкенда; если его нет — формирует понятное
 * русское сообщение по HTTP-статусу.
 */
async function throwHttpError(response: Response): Promise<never> {
  const body = await parseBody(response)
  const detail =
    body !== null && typeof body === 'object' && 'detail' in body
      ? (body as { detail: unknown }).detail
      : undefined

  if (typeof detail === 'string' && detail.trim() !== '') {
    throw new ApiError(detail, response.status)
  }

  throw new ApiError(
    `Сервер ответил с ошибкой (HTTP ${response.status}). Попробуйте ещё раз позже.`,
    response.status,
  )
}

/** Приводит ответ GET /api/microcategories к типу Microcategory[]. */
function assertMicrocategories(value: unknown): Microcategory[] {
  if (!Array.isArray(value)) {
    throw new ApiError('Некорректный ответ сервиса: ожидался список микрокатегорий.')
  }

  return value.map((item) => {
    if (
      item === null ||
      typeof item !== 'object' ||
      typeof (item as Microcategory).mcId !== 'number' ||
      typeof (item as Microcategory).title !== 'string'
    ) {
      throw new ApiError('Некорректный ответ сервиса: повреждена запись микрокатегории.')
    }
    return { mcId: item.mcId, title: item.title }
  })
}

/** Приводит ответ POST /api/drafts к типу DraftsResponse. */
function assertDraftsResponse(value: unknown): DraftsResponse {
  if (
    value === null ||
    typeof value !== 'object' ||
    !Array.isArray((value as DraftsResponse).drafts)
  ) {
    throw new ApiError('Некорректный ответ сервиса: ожидался набор черновиков.')
  }

  const drafts = (value as DraftsResponse).drafts.map((item) => {
    if (item === null || typeof item !== 'object' || typeof item.mcId !== 'number') {
      throw new ApiError('Некорректный ответ сервиса: повреждена запись черновика.')
    }
    return {
      mcId: item.mcId,
      title: typeof item.title === 'string' ? item.title : null,
      text: typeof item.text === 'string' ? item.text : null,
    } satisfies Draft
  })

  return { drafts }
}

/** Приводит ответ GET /api/dataset/random к типу DatasetSample. */
function assertDatasetSample(value: unknown): DatasetSample {
  const sample = value as DatasetSample | null
  if (
    sample === null ||
    typeof sample !== 'object' ||
    typeof sample.description !== 'string' ||
    typeof sample.sourceMcId !== 'number'
  ) {
    throw new ApiError('Некорректный ответ сервиса: ожидалось объявление из датасета.')
  }

  return {
    id: typeof sample.id === 'number' ? sample.id : 0,
    split: sample.split === 'test' ? 'test' : 'train',
    itemId: typeof sample.itemId === 'number' ? sample.itemId : null,
    title: typeof sample.title === 'string' ? sample.title : '',
    description: sample.description,
    sourceMcId: sample.sourceMcId,
    foundMcIds: Array.isArray(sample.foundMcIds) ? sample.foundMcIds : null,
    categories: Array.isArray(sample.categories) ? sample.categories : null,
  }
}

/**
 * GET /api/microcategories — список микрокатегорий категории
 * «Ремонт и отделка».
 */
export async function fetchMicrocategories(): Promise<Microcategory[]> {
  let response: Response
  try {
    response = await fetch('/api/microcategories', {
      headers: { Accept: 'application/json' },
    })
  } catch {
    throw new ApiError('Сервис недоступен. Проверьте, что запущен бэкенд.')
  }

  if (!response.ok) {
    await throwHttpError(response)
  }

  return assertMicrocategories(await parseBody(response))
}

/**
 * GET /api/dataset/random — случайное объявление датасета кейса
 * для подстановки в форму примера.
 */
export async function fetchRandomSample(split: DatasetSplit): Promise<DatasetSample> {
  let response: Response
  try {
    response = await fetch(`/api/dataset/random?split=${split}`, {
      headers: { Accept: 'application/json' },
    })
  } catch {
    throw new ApiError('Сервис недоступен. Проверьте, что запущен бэкенд.')
  }

  if (!response.ok) {
    await throwHttpError(response)
  }

  return assertDatasetSample(await parseBody(response))
}

/**
 * POST /api/drafts — генерация дополнительных черновиков по описанию и
 * исходной микрокатегории.
 */
export async function fetchDrafts(request: DraftsRequest): Promise<DraftsResponse> {
  let response: Response
  try {
    response = await fetch('/api/drafts', {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })
  } catch {
    throw new ApiError('Сервис недоступен. Проверьте, что запущен бэкенд.')
  }

  if (!response.ok) {
    await throwHttpError(response)
  }

  return assertDraftsResponse(await parseBody(response))
}
