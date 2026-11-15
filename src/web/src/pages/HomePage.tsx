import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  Center,
  Container,
  SegmentedControl,
  SimpleGrid,
  Skeleton,
  Stack,
  Text,
  Title,
} from '@mantine/core'
import {
  ApiError,
  fetchDrafts,
  fetchMicrocategories,
  isAbortError,
  streamBatchDrafts,
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

/** Скелетон на время генерации: три заготовки карточек с фирменным знаком. */
function ResultsSkeleton() {
  return (
    <Stack gap="md">
      <Stack align="center" gap={4} py="xs">
        <Box className="avito-pulse">
          <AvitoLogo height={30} markOnly alt="" />
        </Box>
        <Text fw={600} size="lg">
          Подбираем черновики…
        </Text>
      </Stack>

      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md" verticalSpacing="md">
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

export function HomePage() {
  const [microcategories, setMicrocategories] = useState<Microcategory[]>([])
  const [categoriesLoading, setCategoriesLoading] = useState(true)
  const [categoriesError, setCategoriesError] = useState<string | null>(null)

  const [status, setStatus] = useState<Status>('idle')
  const [drafts, setDrafts] = useState<Draft[]>([])
  const [requestError, setRequestError] = useState<string | null>(null)
  const [sourceMcId, setSourceMcId] = useState(0)
  const [lastRequest, setLastRequest] = useState<DraftsRequest | null>(null)
  const [adSelection, setAdSelection] = useState<{ ad: ImportedAd; id: number } | null>(null)
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

  const handleGenerate = useCallback(async (request: DraftsRequest) => {
    setLastRequest(request)
    setStatus('loading')
    setRequestError(null)
    setSourceMcId(request.mcId)
    try {
      const response = await fetchDrafts(request)
      setDrafts(response.drafts)
      setStatus(response.drafts.length === 0 ? 'empty' : 'success')
    } catch (error: unknown) {
      setDrafts([])
      setRequestError(messageOf(error))
      setStatus('error')
    }
  }, [])

  // «Повторить» в алерте ошибки повторяет последний отправленный запрос.
  const retryLastRequest = useCallback(() => {
    if (lastRequest !== null) {
      void handleGenerate(lastRequest)
    }
  }, [lastRequest, handleGenerate])

  const handleSelectAd = useCallback((ad: ImportedAd) => {
    setAdSelection((current) => ({ ad, id: (current?.id ?? 0) + 1 }))
    setInputMode('manual')
  }, [])

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

  // При уходе со страницы прерываем незавершённый поток, чтобы не
  // обновлять состояние уже размонтированного компонента.
  useEffect(() => {
    return () => {
      bulkRunIdRef.current += 1
      bulkAbortRef.current?.abort()
    }
  }, [])

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', minBlockSize: '100vh' }}>
      <AppHeader />

      <Container size="lg" py="xl" style={{ flex: 1 }}>
        <Stack gap="xl">
          <PageHeader />

          <Card withBorder shadow="xs" padding="lg" radius="md" bg="white">
            <Stack gap="md">
              <Title order={2} size="h3">
                Исходное объявление
              </Title>

              <SegmentedControl
                aria-label="Режим ввода объявления"
                data={INPUT_MODE_OPTIONS}
                value={inputMode}
                onChange={(value) => setInputMode(value as InputMode)}
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
                        key={adSelection?.id ?? 'empty'}
                        microcategories={microcategories}
                        loading={status === 'loading'}
                        disabled={categoriesError !== null}
                        selectedAd={adSelection?.ad ?? null}
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
                    microcategories={microcategories}
                    onSelect={handleSelectAd}
                    onProcessAll={startBulk}
                    bulkRunning={bulkAds !== null && bulkPhase === 'running'}
                  />
                </Box>
              </Box>
            </Stack>
          </Card>

          {/* Пакетные результаты: видны независимо от режима ввода. */}
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
          <Box component="section" aria-live="polite" aria-label="Результаты подбора черновиков">
            {status === 'loading' && <ResultsSkeleton />}

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

            {status === 'success' && (
              <ResultsList
                drafts={drafts}
                microcategories={microcategories}
                sourceMcId={sourceMcId}
              />
            )}
          </Box>
        </Stack>
      </Container>

      <AppFooter />
    </Box>
  )
}
