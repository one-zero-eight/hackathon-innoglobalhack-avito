/** Микрокатегория из GET /api/microcategories. */
export interface Microcategory {
  mcId: number
  title: string
}

/** Черновик объявления из SSE-потока POST /api/drafts/stream. */
export interface Draft {
  mcId: number
  title: string | null
  text: string | null
}

/** Тело запроса POST /api/drafts и POST /api/drafts/stream. */
export interface DraftsRequest {
  description: string
  mcId: number
}

/** Объявление пакетного запроса POST /api/drafts/batch (как в импорте таблицы). */
export interface BatchAd {
  description: string
  mcId: number
}

/** Кадр `start`: начало пакетной обработки. */
export interface BatchStartEvent {
  total: number
  concurrency: number
}

/** Кадр `result`: готовый результат по строке входного массива. */
export interface BatchResultEvent {
  index: number
  mcId: number
  drafts: Draft[]
}

/** Кадр `error`: ошибка обработки строки входного массива. */
export interface BatchErrorEvent {
  index: number
  detail: string
}

/** Кадр `done`: итог пакетной обработки. */
export interface BatchDoneEvent {
  processed: number
  succeeded: number
  failed: number
}

/** Кадр `categories` потока POST /api/drafts/stream: цели после классификации. */
export interface StreamCategoriesEvent {
  mcIds: number[]
}

/** Кадр `error` потока POST /api/drafts/stream: безопасный detail бэкенда. */
export interface StreamErrorEvent {
  detail: string
}

/** Обработчики кадров SSE-потока POST /api/drafts/batch. */
export interface BatchStreamHandlers {
  onStart?: (event: BatchStartEvent) => void
  onResult?: (event: BatchResultEvent) => void
  onError?: (event: BatchErrorEvent) => void
  onDone?: (event: BatchDoneEvent) => void
}

/** Обработчики кадров SSE-потока POST /api/drafts/stream. */
export interface DraftsStreamHandlers {
  onCategories?: (event: StreamCategoriesEvent) => void
  onDraft?: (draft: Draft) => void
  onDone?: () => void
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
async function throwHttpError(response: Response, fallbackMessage?: string): Promise<never> {
  const body = await parseBody(response)
  const detail =
    body !== null && typeof body === 'object' && 'detail' in body
      ? (body as { detail: unknown }).detail
      : undefined

  if (typeof detail === 'string' && detail.trim() !== '') {
    throw new ApiError(detail, response.status)
  }

  throw new ApiError(
    fallbackMessage ??
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

/** Приводит одну запись к типу Draft; повреждённая запись — явная ошибка. */
function assertDraftItem(item: unknown): Draft {
  if (item === null || typeof item !== 'object' || typeof (item as Draft).mcId !== 'number') {
    throw new ApiError('Некорректный ответ сервиса: повреждена запись черновика.')
  }
  return {
    mcId: (item as Draft).mcId,
    title: typeof (item as Draft).title === 'string' ? (item as Draft).title : null,
    text: typeof (item as Draft).text === 'string' ? (item as Draft).text : null,
  } satisfies Draft
}

/** Приводит массив черновиков к типу Draft[]. */
function assertDraftsArray(value: unknown): Draft[] {
  if (!Array.isArray(value)) {
    throw new ApiError('Некорректный ответ сервиса: ожидался набор черновиков.')
  }
  return value.map(assertDraftItem)
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
 * POST /api/drafts/stream — потоковая генерация черновиков одного
 * объявления. Ответ — Server-Sent Events: кадр `categories` с целевыми
 * микрокатегориями, затем кадры `draft` по мере готовности каждого
 * черновика и финальный `done`.
 */
export async function streamDrafts(
  request: DraftsRequest,
  handlers: DraftsStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response
  try {
    response = await fetch('/api/drafts/stream', {
      method: 'POST',
      headers: {
        Accept: 'text/event-stream',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal,
    })
  } catch (error: unknown) {
    if (isAbortError(error)) {
      throw error
    }
    throw new ApiError('Сервис недоступен. Проверьте, что запущен бэкенд.')
  }

  if (!response.ok) {
    await throwHttpError(response)
  }

  let finished = false
  let streamError: ApiError | null = null
  await consumeSseStream(response, (frame) => {
    switch (frame.event) {
      case 'categories': {
        const event = assertStreamCategories(parseEventData(frame))
        handlers.onCategories?.(event)
        break
      }
      case 'draft': {
        const draft = assertDraftItem(parseEventData(frame))
        handlers.onDraft?.(draft)
        break
      }
      case 'done': {
        finished = true
        handlers.onDone?.()
        break
      }
      case 'error': {
        const event = assertStreamError(parseEventData(frame))
        streamError = new ApiError(event.detail)
        break
      }
      default:
        break
    }
  })

  if (streamError !== null) {
    throw streamError
  }
  if (!finished) {
    throw new ApiError('Поток ответа прервался. Проверьте, что запущен бэкенд.')
  }
}

/** Один кадр Server-Sent Events: тип события и содержимое `data`. */
export interface SseFrame {
  event: string
  data: string
}

/**
 * Разбирает накопленный текст SSE на готовые кадры: кадр завершается пустой
 * строкой (`\n\n` или `\r\n\r\n`). Возвращает принятые кадры и остаток
 * буфера — неполный кадр, который продолжит накапливаться при следующем чтении.
 */
export function parseSseFrames(buffer: string): { frames: SseFrame[]; rest: string } {
  const frames: SseFrame[] = []
  let rest = buffer

  for (;;) {
    const match = /\r?\n\r?\n/.exec(rest)
    if (match === null) {
      break
    }
    const raw = rest.slice(0, match.index)
    rest = rest.slice(match.index + match[0].length)
    const frame = parseSseFrame(raw)
    if (frame !== null) {
      frames.push(frame)
    }
  }

  return { frames, rest }
}

/** Разбирает один сырой кадр; кадры без `data` (например, комментарии) — null. */
function parseSseFrame(raw: string): SseFrame | null {
  let event = 'message'
  const dataLines: string[] = []

  for (const line of raw.split(/\r?\n/)) {
    if (line === '' || line.startsWith(':')) {
      continue
    }
    const colonIndex = line.indexOf(':')
    const field = colonIndex === -1 ? line : line.slice(0, colonIndex)
    let value = colonIndex === -1 ? '' : line.slice(colonIndex + 1)
    if (value.startsWith(' ')) {
      value = value.slice(1)
    }
    if (field === 'event') {
      event = value
    } else if (field === 'data') {
      dataLines.push(value)
    }
  }

  if (dataLines.length === 0) {
    return null
  }
  return { event, data: dataLines.join('\n') }
}

/** Разбирает JSON из `data` кадра; сбой — явная ошибка протокола. */
function parseEventData(frame: SseFrame): unknown {
  try {
    return JSON.parse(frame.data)
  } catch {
    throw new ApiError(`Сервис прислал некорректные данные события «${frame.event}».`)
  }
}

/** Приводит кадр `start` к типу BatchStartEvent. */
function assertBatchStart(value: unknown): BatchStartEvent {
  const record = value as BatchStartEvent | null
  if (
    record === null ||
    typeof record !== 'object' ||
    typeof record.total !== 'number' ||
    typeof record.concurrency !== 'number'
  ) {
    throw new ApiError('Некорректный ответ сервиса: повреждён кадр start.')
  }
  return { total: record.total, concurrency: record.concurrency }
}

/** Приводит кадр `result` к типу BatchResultEvent. */
function assertBatchResult(value: unknown): BatchResultEvent {
  const record = value as BatchResultEvent | null
  if (
    record === null ||
    typeof record !== 'object' ||
    typeof record.index !== 'number' ||
    typeof record.mcId !== 'number'
  ) {
    throw new ApiError('Некорректный ответ сервиса: повреждён кадр result.')
  }
  return { index: record.index, mcId: record.mcId, drafts: assertDraftsArray(record.drafts) }
}

/** Приводит кадр `error` к типу BatchErrorEvent. */
function assertBatchError(value: unknown): BatchErrorEvent {
  const record = value as BatchErrorEvent | null
  if (
    record === null ||
    typeof record !== 'object' ||
    typeof record.index !== 'number' ||
    typeof record.detail !== 'string'
  ) {
    throw new ApiError('Некорректный ответ сервиса: повреждён кадр error.')
  }
  return { index: record.index, detail: record.detail }
}

/** Приводит кадр `done` к типу BatchDoneEvent. */
function assertBatchDone(value: unknown): BatchDoneEvent {
  const record = value as BatchDoneEvent | null
  if (
    record === null ||
    typeof record !== 'object' ||
    typeof record.processed !== 'number' ||
    typeof record.succeeded !== 'number' ||
    typeof record.failed !== 'number'
  ) {
    throw new ApiError('Некорректный ответ сервиса: повреждён кадр done.')
  }
  return {
    processed: record.processed,
    succeeded: record.succeeded,
    failed: record.failed,
  }
}

/** Приводит кадр `categories` к типу StreamCategoriesEvent. */
function assertStreamCategories(value: unknown): StreamCategoriesEvent {
  const record = value as StreamCategoriesEvent | null
  if (
    record === null ||
    typeof record !== 'object' ||
    !Array.isArray(record.mcIds) ||
    record.mcIds.some((mcId) => typeof mcId !== 'number')
  ) {
    throw new ApiError('Некорректный ответ сервиса: повреждён кадр categories.')
  }
  return { mcIds: record.mcIds }
}

/** Приводит кадр `error` потока к типу StreamErrorEvent. */
function assertStreamError(value: unknown): StreamErrorEvent {
  const record = value as StreamErrorEvent | null
  if (
    record === null ||
    typeof record !== 'object' ||
    typeof record.detail !== 'string'
  ) {
    throw new ApiError('Некорректный ответ сервиса: повреждён кадр error.')
  }
  return { detail: record.detail }
}

/** Отправляет один кадр потока в обработчики; неизвестные события игнорируются. */
function dispatchBatchFrame(frame: SseFrame, handlers: BatchStreamHandlers): void {
  // Кадр валидируем до вызова обработчика: при отсутствии обработчика
  // optional-call не вычислил бы аргументы и протоколная ошибка прошла бы мимо.
  switch (frame.event) {
    case 'start': {
      const event = assertBatchStart(parseEventData(frame))
      handlers.onStart?.(event)
      break
    }
    case 'result': {
      const event = assertBatchResult(parseEventData(frame))
      handlers.onResult?.(event)
      break
    }
    case 'error': {
      const event = assertBatchError(parseEventData(frame))
      handlers.onError?.(event)
      break
    }
    case 'done': {
      const event = assertBatchDone(parseEventData(frame))
      handlers.onDone?.(event)
      break
    }
    default:
      break
  }
}

/** Отменённый ли это отказ браузера (AbortController) или сетевая ошибка. */
export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

/**
 * Читает тело SSE-ответа через getReader(), разбирает кадры по мере
 * готовности и передаёт каждый в onFrame. Сетевые сбои превращаются в
 * ApiError с русским сообщением; отмена через `signal` читателя
 * пробрасывается как есть (AbortError). Хвост без завершающей пустой
 * строки добирается в конце чтения.
 */
async function consumeSseStream(
  response: Response,
  onFrame: (frame: SseFrame) => void,
): Promise<void> {
  if (response.body === null) {
    throw new ApiError('Сервер не передал поток ответа. Попробуйте ещё раз.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }
      buffer += decoder.decode(value, { stream: true })
      const { frames, rest } = parseSseFrames(buffer)
      buffer = rest
      for (const frame of frames) {
        onFrame(frame)
      }
    }

    // Хвост без завершающей пустой строки: добираем остаток буфера.
    buffer += decoder.decode()
    if (buffer.trim() !== '') {
      const { frames } = parseSseFrames(`${buffer}\n\n`)
      for (const frame of frames) {
        onFrame(frame)
      }
    }
  } catch (error: unknown) {
    if (!isAbortError(error)) {
      // Ошибка протокола или обрыв потока — закрываем чтение, не оставляем lock.
      await reader.cancel().catch(() => undefined)
    }
    if (error instanceof ApiError || isAbortError(error)) {
      throw error
    }
    throw new ApiError('Поток ответа прервался. Проверьте, что запущен бэкенд.')
  }
}

/**
 * POST /api/drafts/batch — пакетная генерация черновиков с потоковым
 * ответом Server-Sent Events. Разбирает кадры по мере готовности и
 * вызывает обработчики onStart/onResult/onError/onDone. Сетевые сбои и
 * не-2xx превращаются в ApiError с русским сообщением; отмена через
 * `signal` пробрасывается как есть (AbortError).
 */
export async function streamBatchDrafts(
  ads: BatchAd[],
  concurrency: number | null,
  handlers: BatchStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response
  try {
    response = await fetch('/api/drafts/batch', {
      method: 'POST',
      headers: {
        Accept: 'text/event-stream',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ads, concurrency }),
      signal,
    })
  } catch (error: unknown) {
    if (isAbortError(error)) {
      throw error
    }
    throw new ApiError('Сервис недоступен. Проверьте, что запущен бэкенд.')
  }

  if (!response.ok) {
    await throwHttpError(
      response,
      response.status === 422
        ? 'Сервер отклонил пакетный запрос (HTTP 422): проверьте, что файл содержит только description и mcId.'
        : undefined,
    )
  }

  await consumeSseStream(response, (frame) => dispatchBatchFrame(frame, handlers))
}
