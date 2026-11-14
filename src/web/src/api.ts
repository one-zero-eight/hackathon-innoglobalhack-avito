import type { MantineColor } from '@mantine/core'

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

/** Ошибка обращения к бэкенду с читаемым русским сообщением. */
export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status = 0) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/**
 * Акцент для подписи черновика: если целевая микрокатегория совпала с
 * исходной (не должно), карточка подсвечивается красным.
 */
export function draftAccent(targetMcId: number, sourceMcId: number): MantineColor {
  return targetMcId === sourceMcId ? 'red' : 'avito'
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
