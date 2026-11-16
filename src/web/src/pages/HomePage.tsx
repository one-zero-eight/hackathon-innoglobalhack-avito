import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  Center,
  Container,
  Grid,
  Progress,
  SegmentedControl,
  SimpleGrid,
  Skeleton,
  Stack,
  Text,
  Title,
} from '@mantine/core'
import {
  ApiError,
  fetchMicrocategories,
  isAbortError,
  streamBatchDrafts,
  streamDrafts,
  type BatchDoneEvent,
  type Draft,
  type DraftsRequest,
  type Microcategory,
} from '../api'
import type { ImportedAd } from '../adsParser'
import { createBulkRows, type BulkPhase, type BulkRow } from '../bulkTypes'
import { AdsDropzone } from '../components/AdsDropzone'
import { AppFooter } from '../components/AppFooter'
import { AppHeader } from '../components/AppHeader'
import { AvitoLogo } from '../components/AvitoLogo'
import { BulkResults } from '../components/BulkResults'
import { DraftsForm } from '../components/DraftsForm'
import { EmptyState } from '../components/EmptyState'
import { AlertIcon, RefreshIcon } from '../components/icons'
import { ResultsList } from '../components/ResultsList'

/** Состояния, в которых может находиться страница после запроса. */
type Status = 'idle' | 'loading' | 'error' | 'success' | 'empty'
type InputMode = 'manual' | 'table'

const INPUT_MODE_OPTIONS = [
  { value: 'manual' as const, label: 'Ручной ввод' },
  { value: 'table' as const, label: 'Загрузка таблицы' },
]

function messageOf(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message
  }
  if (error instanceof Error && error.message) {
    return error.message
  }
  return 'Неизвестная ошибка. Попробуйте ещё раз.'
}

/** Скелетон до списка категорий: три заготовки карточек черновиков. */
function DraftsSkeletonGrid() {
  return (
    <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md" verticalSpacing="md">
      {[0, 1, 2].map((key) => (
        <Card key={key} withBorder shadow="xs" padding="md" radius="md" bg="white">
          <Stack gap="sm">
            <Skeleton height={22} width={180} radius="md" />
            <Skeleton height={18} radius="md" />
            <Skeleton height={14} radius="md" />
            <Skeleton height={14} width="80%" radius="md" />
          </Stack>
        </Card>
      ))}
    </SimpleGrid>
  )
}

/** Статус и полоса прогресса ручной генерации по кадрам потока. */
function GenerationProgress({ done, total }: { done: number; total: number | null }) {
  const percent = total !== null && total > 0 ? (done / total) * 100 : 100
  const label =
    total === null
      ? 'Определяем подходящие микрокатегории…'
      : total === 0
        ? 'Подходящие микрокатегории не найдены'
        : `Генерируем черновик ${Math.min(done + 1, total)} из ${total}…`

  return (
    <Stack gap="md">
      <Stack align="center" gap={4} py="xs">
        <Box className="avito-pulse">
          <AvitoLogo height={30} markOnly alt="" />
        </Box>
        <Text fw={600} size="lg" ta="center">
          {label}
        </Text>
      </Stack>

      <Progress
        value={percent}
        animated
        size="lg"
        radius="xl"
        aria-label={
          total === null || total === 0
            ? 'Определяем подходящие микрокатегории'
            : `Сгенерировано ${done} из ${total} черновиков`
        }
      />
    </Stack>
  )
}

/** Заголовок страницы в фирменном стиле avito.ru: крупный, с синим акцентом. */
function PageHeader() {
  return (
    <Stack gap="xs">
      <Title
        order={1}
        fw={800}
        lh={1.1}
        size="clamp(2rem, 4vw, 2.75rem)"
        className="text-balance"
      >
        Дополнительные{' '}
        <Box component="span" c="avito.6">
          черновики
        </Box>{' '}
        объявлений
      </Title>
      <Text c="dimmed" size="md" maw={640}>
        Опишите услугу и укажите текущую категорию — сервис найдёт работы в других
        микрокатегориях раздела «Ремонт и отделка» и соберёт готовые к публикации
        черновики.
      </Text>
    </Stack>
  )
}

/**
 * Заглушка правой колонки до первого запроса: объясняет, что здесь появятся
 * черновики, чтобы колонка не выглядела сломанной и пустой.
 */
function ResultsPlaceholder() {
  return (
    <Card withBorder shadow="xs" padding="xl" radius="md" bg="white">
      <Stack align="center" justify="center" gap="xs" mih={140}>
        <Box className="avito-pulse" aria-hidden>
          <AvitoLogo height={34} markOnly alt="" />
        </Box>
        <Text fw={600} ta="center">
          Здесь появятся черновики
        </Text>
        <Text c="dimmed" size="sm" ta="center" maw={420}>
          Опишите услугу слева и запустите подбор — результаты появятся в этой
          колонке.
        </Text>
      </Stack>
    </Card>
  )
}

export function HomePage() {
  const [microcategories, setMicrocategories] = useState<Microcategory[]>([])
  const [categoriesLoading, setCategoriesLoading] = useState(true)
  const [categoriesError, setCategoriesError] = useState<string | null>(null)

  const [status, setStatus] = useState<Status>('idle')
  const [drafts, setDrafts] = useState<Draft[]>([])
  const [requestError, setRequestError] = useState<string | null>(null)
  const [sourceMcId, setSourceMcId] = useState(0)
  const [lastRequest, setLastRequest] = useState<DraftsRequest | null>(null)
  /** Число целевых категорий ручного запроса (из кадра categories). */
  const [manualTotal, setManualTotal] = useState<number | null>(null)
  const [inputMode, setInputMode] = useState<InputMode>('manual')
  const [inputPanelHeight, setInputPanelHeight] = useState<number | null>(null)

  // Пакетная обработка импортированной таблицы (POST /api/drafts/batch, SSE).
  const [bulkAds, setBulkAds] = useState<ImportedAd[] | null>(null)
  const [bulkRows, setBulkRows] = useState<BulkRow[]>([])
  const [bulkPhase, setBulkPhase] = useState<BulkPhase>('running')
  const [bulkTotal, setBulkTotal] = useState(0)
  const [bulkSummary, setBulkSummary] = useState<BatchDoneEvent | null>(null)
  const [bulkError, setBulkError] = useState<string | null>(null)

  const categoriesRequestId = useRef(0)
  const manualPanelRef = useRef<HTMLDivElement>(null)
  const tablePanelRef = useRef<HTMLDivElement>(null)
  const manualAbortRef = useRef<AbortController | null>(null)
  const manualRunIdRef = useRef(0)
  const bulkAbortRef = useRef<AbortController | null>(null)
  const bulkRunIdRef = useRef(0)

  // Загрузка списка микрокатегорий; повтор вызова отменяет предыдущий ответ.
  // Состояния меняются только в промисах, без синхронных setState в эффекте.
  const loadCategories = useCallback(() => {
    const requestId = ++categoriesRequestId.current

    fetchMicrocategories()
      .then((items) => {
        if (requestId === categoriesRequestId.current) {
          setMicrocategories(items)
          setCategoriesError(null)
        }
      })
      .catch((error: unknown) => {
        if (requestId === categoriesRequestId.current) {
          setCategoriesError(messageOf(error))
        }
      })
      .finally(() => {
        if (requestId === categoriesRequestId.current) {
          setCategoriesLoading(false)
        }
      })
  }, [])

  // «Повторить» в алерте: заново запрашивает список микрокатегорий.
  const retryCategories = useCallback(() => {
    setCategoriesLoading(true)
    setCategoriesError(null)
    loadCategories()
  }, [loadCategories])

  useEffect(() => {
    loadCategories()
  }, [loadCategories])

  useLayoutEffect(() => {
    const panel = inputMode === 'manual' ? manualPanelRef.current : tablePanelRef.current
    if (panel === null) {
      return
    }

    const updateHeight = () => setInputPanelHeight(panel.scrollHeight)
    updateHeight()

    const observer = new ResizeObserver(updateHeight)
    observer.observe(panel)
    return () => observer.disconnect()
  }, [inputMode, categoriesLoading, categoriesError])

  /**
   * Ручная генерация через SSE-поток POST /api/drafts/stream: кадр
   * `categories` даёт общее число черновиков, каждый кадр `draft` добавляет
   * карточку сразу по готовности. Каждый запуск получает runId; устаревшие
   * потоки игнорируются, новая отправка отменяет предыдущую.
   */
  const handleGenerate = useCallback(async (request: DraftsRequest) => {
    const runId = ++manualRunIdRef.current
    manualAbortRef.current?.abort()
    const controller = new AbortController()
    manualAbortRef.current = controller

    let received = 0

    setLastRequest(request)
    setStatus('loading')
    setRequestError(null)
    setDrafts([])
    setManualTotal(null)
    setSourceMcId(request.mcId)

    try {
      await streamDrafts(
        request,
        {
          onCategories: (event) => {
            if (runId === manualRunIdRef.current) {
              setManualTotal(event.mcIds.length)
            }
          },
          onDraft: (draft) => {
            if (runId !== manualRunIdRef.current) {
              return
            }
            received += 1
            setDrafts((current) => [...current, draft])
          },
          onDone: () => {
            if (runId !== manualRunIdRef.current) {
              return
            }
            setStatus(received === 0 ? 'empty' : 'success')
          },
        },
        controller.signal,
      )
    } catch (error: unknown) {
      if (runId !== manualRunIdRef.current) {
        return
      }
      if (isAbortError(error)) {
        setStatus('idle')
        setDrafts([])
        setManualTotal(null)
        return
      }
      setDrafts([])
      setManualTotal(null)
      setRequestError(messageOf(error))
      setStatus('error')
    } finally {
      if (runId === manualRunIdRef.current) {
        manualAbortRef.current = null
      }
    }
  }, [])

  // «Повторить» в алерте ошибки повторяет последний отправленный запрос.
  const retryLastRequest = useCallback(() => {
    if (lastRequest !== null) {
      void handleGenerate(lastRequest)
    }
  }, [lastRequest, handleGenerate])

  /** Сброс ручного результата и остановка его потока (при смене режима). */
  const resetManualResults = useCallback(() => {
    manualRunIdRef.current += 1
    manualAbortRef.current?.abort()
    manualAbortRef.current = null
    setStatus('idle')
    setDrafts([])
    setRequestError(null)
    setManualTotal(null)
  }, [])

  /** Сброс пакетного результата и отмена его потока (при смене режима). */
  const resetBulkResults = useCallback(() => {
    bulkRunIdRef.current += 1
    bulkAbortRef.current?.abort()
    bulkAbortRef.current = null
    setBulkAds(null)
    setBulkRows([])
    setBulkPhase('running')
    setBulkTotal(0)
    setBulkSummary(null)
    setBulkError(null)
  }, [])

  /**
   * Смена режима ввода: результаты противоположного режима скрываются —
   * панель CSV не висит над ручным вводом, и наоборот.
   */
  const handleInputModeChange = useCallback(
    (value: string) => {
      const nextMode = value as InputMode
      if (nextMode === inputMode) {
        return
      }
      if (nextMode === 'manual') {
        resetBulkResults()
      } else {
        resetManualResults()
      }
      setInputMode(nextMode)
    },
    [inputMode, resetBulkResults, resetManualResults],
  )

  /**
   * Запуск пакетной обработки: отправляет весь массив на POST /api/drafts/batch
   * и ведёт состояние строк по кадрам SSE. Каждый запуск получает runId;
   * устаревшие потоки игнорируются, а новая задача отменяет предыдущую.
   */
  const startBulk = useCallback((ads: ImportedAd[], concurrency: number) => {
    const runId = ++bulkRunIdRef.current
    bulkAbortRef.current?.abort()
    const controller = new AbortController()
    bulkAbortRef.current = controller

    setBulkAds(ads)
    setBulkRows(createBulkRows(ads.length))
    setBulkPhase('running')
    setBulkTotal(ads.length)
    setBulkSummary(null)
    setBulkError(null)

    // Строку закрываем результатом/ошибкой и берём в работу следующую
    // ожидающую: так число «в работе» примерно равно параллелизму.
    const settle = (index: number, patch: BulkRow) =>
      setBulkRows((rows) => {
        if (index < 0 || index >= rows.length) {
          return rows
        }
        const next = rows.map((row, rowIndex) => (rowIndex === index ? patch : row))
        const nextPending = next.findIndex((row) => row.status === 'pending')
        if (nextPending !== -1) {
          next[nextPending] = { ...next[nextPending], status: 'in_progress' }
        }
        return next
      })

    void streamBatchDrafts(
      ads,
      concurrency,
      {
        onStart: (event) => {
          if (runId !== bulkRunIdRef.current) {
            return
          }
          setBulkTotal(event.total)
          setBulkRows((rows) => {
            const active = Math.min(Math.max(event.concurrency, 1), rows.length)
            return rows.map((row, index) =>
              index < active ? { ...row, status: 'in_progress' } : row,
            )
          })
        },
        onResult: (event) => {
          if (runId !== bulkRunIdRef.current) {
            return
          }
          settle(event.index, { status: 'done', drafts: event.drafts, error: null })
        },
        onError: (event) => {
          if (runId !== bulkRunIdRef.current) {
            return
          }
          settle(event.index, { status: 'error', drafts: [], error: event.detail })
        },
        onDone: (event) => {
          if (runId !== bulkRunIdRef.current) {
            return
          }
          setBulkSummary(event)
          setBulkPhase('done')
        },
      },
      controller.signal,
    )
      .catch((error: unknown) => {
        if (runId !== bulkRunIdRef.current) {
          return
        }
        if (isAbortError(error)) {
          setBulkRows((rows) =>
            rows.map((row) =>
              row.status === 'in_progress' ? { ...row, status: 'pending' } : row,
            ),
          )
          setBulkPhase('cancelled')
          return
        }
        setBulkError(messageOf(error))
        setBulkPhase('failed')
      })
      .finally(() => {
        if (runId === bulkRunIdRef.current) {
          bulkAbortRef.current = null
        }
      })
  }, [])

  /** Остановка пакетной обработки через AbortController. */
  const cancelBulk = useCallback(() => {
    bulkAbortRef.current?.abort()
  }, [])

  // При уходе со страницы прерываем незавершённые потоки, чтобы не
  // обновлять состояние уже размонтированного компонента.
  useEffect(() => {
    return () => {
      manualRunIdRef.current += 1
      manualAbortRef.current?.abort()
      bulkRunIdRef.current += 1
      bulkAbortRef.current?.abort()
    }
  }, [])

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', minBlockSize: '100vh' }}>
      <AppHeader />

      <Container size="xl" py="xl" style={{ flex: 1 }}>
        <Stack gap="xl">
          <PageHeader />

          {/* Две колонки на широких экранах: ввод слева, результаты справа.
              На узких Grid схлопывается в одну колонку: сначала ввод,
              затем результаты — порядок dom-дочерей сохранён. */}
          <Grid gap="lg">
            <Grid.Col span={{ base: 12, lg: 5 }}>
              {/* Sticky-обёртка: при прокрутке длинных результатов форма
                  ввода остаётся видимой (только от lg, см. index.css). */}
              <Box className="input-column-sticky">
                <Card withBorder shadow="xs" padding="lg" radius="md" bg="white">
                  <Stack gap="md">
                    <Title order={2} size="h3">
                      Исходное объявление
                    </Title>

                    <SegmentedControl
                      aria-label="Режим ввода объявления"
                      data={INPUT_MODE_OPTIONS}
                      value={inputMode}
                      onChange={handleInputModeChange}
                      fullWidth
                      maw={480}
                      mx="auto"
                    />

                    <Box
                      className="input-panels-viewport"
                      style={{ height: inputPanelHeight ?? 'auto' }}
                    >
                      <Box
                        ref={manualPanelRef}
                        className="input-panel"
                        hidden={inputMode !== 'manual'}
                      >
                        <Stack gap="md">
                          {categoriesError !== null && (
                            <Alert
                              color="red"
                              title="Не удалось загрузить микрокатегории"
                              radius="md"
                              icon={<AlertIcon />}
                            >
                              <Stack gap="sm">
                                <Text size="sm">{categoriesError}</Text>
                                <Button
                                  variant="light"
                                  color="red"
                                  size="sm"
                                  leftSection={<RefreshIcon size={16} />}
                                  onClick={retryCategories}
                                  w="fit-content"
                                >
                                  Повторить
                                </Button>
                              </Stack>
                            </Alert>
                          )}

                          {categoriesLoading ? (
                            <Center py="lg">
                              <Stack align="center" gap={6}>
                                <Box className="avito-pulse">
                                  <AvitoLogo height={34} markOnly alt="" />
                                </Box>
                                <Text c="dimmed" size="sm">
                                  Загружаем микрокатегории…
                                </Text>
                              </Stack>
                            </Center>
                          ) : (
                            <DraftsForm
                              microcategories={microcategories}
                              loading={status === 'loading'}
                              disabled={categoriesError !== null}
                              onSubmit={handleGenerate}
                            />
                          )}
                        </Stack>
                      </Box>

                      <Box
                        ref={tablePanelRef}
                        className="input-panel"
                        hidden={inputMode !== 'table'}
                      >
                        <AdsDropzone
                          onProcessAll={startBulk}
                          bulkRunning={bulkAds !== null && bulkPhase === 'running'}
                        />
                      </Box>
                    </Box>
                  </Stack>
                </Card>
              </Box>
            </Grid.Col>

            <Grid.Col span={{ base: 12, lg: 7 }}>
              <Stack gap="lg">
                {/* Пакетные результаты: видны в режиме таблицы; переключение
                    на ручной ввод скрывает панель и отменяет поток. */}
                {bulkAds !== null && (
                  <Box
                    component="section"
                    aria-live="polite"
                    aria-label="Результаты пакетной обработки"
                  >
                    <Card withBorder shadow="xs" padding="lg" radius="md" bg="white">
                      <BulkResults
                        ads={bulkAds}
                        rows={bulkRows}
                        phase={bulkPhase}
                        total={bulkTotal}
                        summary={bulkSummary}
                        globalError={bulkError}
                        microcategories={microcategories}
                        onCancel={cancelBulk}
                      />
                    </Card>
                  </Box>
                )}

                {/* Зона результатов: изменения озвучиваются скринридером. */}
                <Box
                  component="section"
                  aria-live="polite"
                  aria-label="Результаты подбора черновиков"
                >
                  <Stack gap="lg">
                    {status === 'idle' && bulkAds === null && <ResultsPlaceholder />}

                    {status === 'loading' && (
                      <GenerationProgress done={drafts.length} total={manualTotal} />
                    )}

                    {status === 'error' && (
                      <Alert
                        color="red"
                        title="Ошибка подбора черновиков"
                        radius="md"
                        icon={<AlertIcon />}
                      >
                        <Stack gap="sm">
                          <Text size="sm">{requestError}</Text>
                          <Button
                            variant="light"
                            color="red"
                            size="sm"
                            leftSection={<RefreshIcon size={16} />}
                            onClick={retryLastRequest}
                            disabled={lastRequest === null}
                            w="fit-content"
                          >
                            Повторить
                          </Button>
                        </Stack>
                      </Alert>
                    )}

                    {status === 'empty' && <EmptyState />}

                    {/* Черновики ручного запроса: во время генерации карточки
                        приходят по одной, недостающие места занимают
                        заглушки-Skeleton; на success тот же список остаётся
                        на месте, без перемонтирования. */}
                    {(status === 'loading' || status === 'success') &&
                      manualTotal !== null &&
                      manualTotal > 0 && (
                        <ResultsList
                          drafts={drafts}
                          microcategories={microcategories}
                          sourceMcId={sourceMcId}
                          pending={
                            status === 'loading'
                              ? Math.max(manualTotal - drafts.length, 0)
                              : 0
                          }
                        />
                      )}

                    {status === 'loading' && manualTotal === null && (
                      <DraftsSkeletonGrid />
                    )}
                  </Stack>
                </Box>
              </Stack>
            </Grid.Col>
          </Grid>
        </Stack>
      </Container>

      <AppFooter />
    </Box>
  )
}
