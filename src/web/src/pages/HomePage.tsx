import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  Card,
  Center,
  Container,
  Loader,
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
import { DraftsForm } from '../components/DraftsForm'
import { EmptyState } from '../components/EmptyState'
import { ResultsList } from '../components/ResultsList'

/** Состояния, в которых может находиться страница после запроса. */
type Status = 'idle' | 'loading' | 'error' | 'success' | 'empty'

function messageOf(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message
  }
  if (error instanceof Error && error.message) {
    return error.message
  }
  return 'Неизвестная ошибка. Попробуйте ещё раз.'
}

/** Скелетон на время генерации: три заготовки карточек. */
function ResultsSkeleton() {
  return (
    <Stack gap="md">
      <Text fw={600} size="lg">
        Подбираем черновики…
      </Text>
      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md" verticalSpacing="md">
        {[0, 1, 2].map((key) => (
          <Card key={key} withBorder shadow="xs" padding="md" radius="sm" bg="white">
            <Stack gap="sm">
              <Skeleton height={22} width={180} radius="sm" />
              <Skeleton height={18} radius="sm" />
              <Skeleton height={14} radius="sm" />
              <Skeleton height={14} width="80%" radius="sm" />
            </Stack>
          </Card>
        ))}
      </SimpleGrid>
    </Stack>
  )
}

/** Заголовок страницы в фирменном стиле avito.ru. */
function PageHeader() {
  return (
    <Stack gap={2}>
      <Title order={1} fw={700} lh={1.2} className="text-balance">
        Дополнительные черновики объявлений
      </Title>
      <Text c="dimmed" size="sm">
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

  // Подгружаем список микрокатегорий при монтировании. Начальное значение
  // categoriesLoading === true, поэтому синхронный setState в эффекте не нужен.
  useEffect(() => {
    let active = true
    fetchMicrocategories()
      .then((items) => {
        if (active) {
          setMicrocategories(items)
          setCategoriesError(null)
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setCategoriesError(messageOf(error))
        }
      })
      .finally(() => {
        if (active) {
          setCategoriesLoading(false)
        }
      })
    return () => {
      active = false
    }
  }, [])

  const handleGenerate = useCallback(async (request: DraftsRequest) => {
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

  return (
    <Container size="lg" py="xl">
      <Stack gap="xl">
        <PageHeader />

        <Card withBorder shadow="xs" padding="lg" radius="sm" bg="white">
          <Stack gap="md">
            <Title order={2} size="h3">
              Исходное объявление
            </Title>

            {categoriesError !== null && (
              <Alert color="red" title="Не удалось загрузить микрокатегории" radius="sm">
                {categoriesError}
              </Alert>
            )}

            {categoriesLoading ? (
              <Center py="lg">
                <Loader color="avito" />
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
        </Card>

        {status === 'loading' && <ResultsSkeleton />}

        {status === 'error' && (
          <Alert color="red" title="Ошибка подбора черновиков" radius="sm">
            {requestError}
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
      </Stack>
    </Container>
  )
}
