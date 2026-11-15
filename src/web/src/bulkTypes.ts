import type { Draft } from './api'

/** Статус строки входного массива в пакетной обработке. */
export type BulkRowStatus = 'pending' | 'in_progress' | 'done' | 'error'

/** Состояние одной строки: ожидание, в работе, готово или ошибка. */
export interface BulkRow {
  status: BulkRowStatus
  drafts: Draft[]
  error: string | null
}

/** Фаза пакетной задачи: идёт, завершена, остановлена или сорвалась. */
export type BulkPhase = 'running' | 'done' | 'cancelled' | 'failed'

/** Начальное состояние строк: все в очереди на ожидание. */
export function createBulkRows(count: number): BulkRow[] {
  return Array.from({ length: count }, () => ({
    status: 'pending' as const,
    drafts: [],
    error: null,
  }))
}
