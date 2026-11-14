import { useCallback, useEffect, useRef, useState } from 'react'
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
  type Draft,
  type DraftsRequest,
  type Microcategory,
} from '../api'
import type { ImportedAd } from '../adsParser'
import { AdsDropzone } from '../components/AdsDropzone'
import { AppFooter } from '../components/AppFooter'
import { AppHeader } from '../components/AppHeader'
import { AvitoLogo } from '../components/AvitoLogo'
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

  const categoriesRequestId = useRef(0)

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

              <Box className="input-panel" hidden={inputMode !== 'manual'}>
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

              <Box className="input-panel" hidden={inputMode !== 'table'}>
                <AdsDropzone microcategories={microcategories} onSelect={handleSelectAd} />
              </Box>
            </Stack>
          </Card>

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
