import { useCallback, useMemo, useRef, useState, type FormEvent } from 'react'
import {
  ActionIcon,
  Alert,
  Box,
  Button,
  Group,
  Modal,
  Select,
  Stack,
  Text,
  Textarea,
} from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import { fetchRandomSample, type DraftsRequest, type Microcategory } from '../api'
import { AlertIcon, MaximizeIcon, ShuffleIcon } from './icons'

const MIN_DESCRIPTION_LENGTH = 10

/** Подпись и подсказка поля описания — общие для формы и увеличенного редактора. */
const DESCRIPTION_LABEL = 'Описание объявления'
const DESCRIPTION_PLACEHOLDER =
  'Например: выполню ремонт под ключ: штукатурка, плитка, электрика…'

/** Начиная с этой ширины (48em) увеличенный редактор занимает весь экран. */
const FULL_SCREEN_BREAKPOINT = '(max-width: 48em)'

interface DraftsFormProps {
  microcategories: Microcategory[]
  /** Загрузка в процессе генерации черновиков. */
  loading: boolean
  /** Список ещё не загрузился — поля блокируем. */
  disabled?: boolean
  onSubmit: (request: DraftsRequest) => void
}

interface FormErrors {
  description?: string
  mcId?: string
}

function validate(
  description: string,
  mcId: string | null,
  microcategories: Microcategory[],
): FormErrors {
  const errors: FormErrors = {}

  const trimmed = description.trim()
  if (trimmed.length === 0) {
    errors.description = 'Укажите описание объявления.'
  } else if (trimmed.length < MIN_DESCRIPTION_LENGTH) {
    errors.description = `Описание слишком короткое: минимум ${MIN_DESCRIPTION_LENGTH} символов.`
  }

  if (mcId === null || mcId === '') {
    errors.mcId = 'Выберите текущую микрокатегорию.'
  } else if (!microcategories.some((item) => item.mcId === Number(mcId))) {
    errors.mcId = 'Выберите микрокатегорию из справочника.'
  }

  return errors
}

/** Форма подбора черновиков: описание услуги + исходная микрокатегория. */
export function DraftsForm({
  microcategories,
  loading,
  disabled = false,
  onSubmit,
}: DraftsFormProps) {
  const [description, setDescription] = useState('')
  const [mcId, setMcId] = useState<string | null>(null)
  const [errors, setErrors] = useState<FormErrors>({})
  const [sampleLoading, setSampleLoading] = useState(false)
  const [sampleError, setSampleError] = useState<string | null>(null)
  const [expandedOpened, setExpandedOpened] = useState(false)

  const descriptionRef = useRef<HTMLTextAreaElement>(null)
  const mcIdRef = useRef<HTMLInputElement>(null)
  const sampleRequestId = useRef(0)

  // На узких экранах увеличенный редактор открывается на весь экран.
  const fullScreenEditor = useMediaQuery(FULL_SCREEN_BREAKPOINT)

  // Увеличенный редактор описания: открытие и закрытие модалки.
  const openExpanded = useCallback(() => setExpandedOpened(true), [])
  const closeExpanded = useCallback(() => setExpandedOpened(false), [])
  // После анимации выхода фокус возвращаем в inline-Textarea формы.
  const handleExpandedExited = useCallback(() => {
    descriptionRef.current?.focus()
  }, [])

  // Значение Select — строка; подписываем как «101 — Ремонт под ключ».
  const options = useMemo(
    () =>
      microcategories.map((item) => ({
        value: String(item.mcId),
        label: `${item.mcId} — ${item.title}`,
      })),
    [microcategories],
  )

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      const nextErrors = validate(description, mcId, microcategories)
      setErrors(nextErrors)

      // При ошибке валидации фокусируем первое невалидное поле.
      if (nextErrors.description) {
        descriptionRef.current?.focus()
        return
      }
      if (nextErrors.mcId) {
        mcIdRef.current?.focus()
        return
      }

      onSubmit({ description: description.trim(), mcId: Number(mcId) })
    },
    [description, mcId, microcategories, onSubmit],
  )

  // «Пример из датасета»: случайное объявление train-сплита подставляется
  // в форму; исходная категория train-сэмплов всегда 101. Повторный клик
  // отменяет эффект предыдущего ответа через счётчик запросов.
  const handleLoadSample = useCallback(() => {
    const requestId = ++sampleRequestId.current
    setSampleLoading(true)
    setSampleError(null)

    fetchRandomSample('train')
      .then((sample) => {
        if (requestId !== sampleRequestId.current) {
          return
        }
        setDescription(sample.description)
        setMcId(String(sample.sourceMcId))
        setErrors({})
      })
      .catch((error: unknown) => {
        if (requestId !== sampleRequestId.current) {
          return
        }
        const message =
          error instanceof Error && error.message
            ? error.message
            : 'Не удалось загрузить пример из датасета.'
        setSampleError(message)
      })
      .finally(() => {
        if (requestId === sampleRequestId.current) {
          setSampleLoading(false)
        }
      })
  }, [])

  return (
    <form onSubmit={handleSubmit} noValidate>
      <Stack gap="md">
        <Box pos="relative">
          <ActionIcon
            pos="absolute"
            top={0}
            right={0}
            variant="subtle"
            color="gray"
            size="sm"
            aria-label="Развернуть окно ввода описания"
            title="Развернуть окно ввода описания"
            disabled={disabled}
            onClick={openExpanded}
          >
            <MaximizeIcon size={15} />
          </ActionIcon>
          <Textarea
            label={DESCRIPTION_LABEL}
            description={`Минимум ${MIN_DESCRIPTION_LENGTH} символов`}
            placeholder={DESCRIPTION_PLACEHOLDER}
            autosize
            minRows={5}
            maxRows={12}
            value={description}
            onChange={(event) => setDescription(event.currentTarget.value)}
            error={errors.description}
            disabled={disabled}
            withAsterisk
            ref={descriptionRef}
          />
        </Box>

        <Select
          label="Текущая микрокатегория"
          placeholder="Выберите микрокатегорию"
          data={options}
          value={mcId}
          onChange={(value) => setMcId(value)}
          error={errors.mcId}
          disabled={disabled}
          allowDeselect
          clearable
          searchable
          withAsterisk
          nothingFoundMessage="Ничего не найдено"
          ref={mcIdRef}
        />

        <Group>
          <Button type="submit" loading={loading} disabled={disabled}>
            Подобрать черновики
          </Button>
          <Button
            type="button"
            variant="light"
            onClick={handleLoadSample}
            loading={sampleLoading}
            disabled={disabled}
            leftSection={<ShuffleIcon size={16} />}
          >
            Пример из датасета
          </Button>
        </Group>

        {/* Ошибка подстановки примера — ненавязчивый алерт с detail бэкенда. */}
        {sampleError !== null && (
          <Alert
            color="red"
            variant="light"
            radius="md"
            icon={<AlertIcon size={16} />}
            title="Не удалось подставить пример"
            withCloseButton
            onClose={() => setSampleError(null)}
          >
            <Text size="sm">{sampleError}</Text>
          </Alert>
        )}

        {/* Увеличенный редактор описания: крупный Textarea привязан к тому же
            состоянию description, поэтому правки синхронизируются с формой
            в реальном времени. Модалка рендерится в Portal вне <form>, а Enter
            в textarea даёт перенос строки — форма сама по себе не отправляется. */}
        <Modal
          opened={expandedOpened}
          onClose={closeExpanded}
          title={DESCRIPTION_LABEL}
          size="xl"
          centered
          fullScreen={fullScreenEditor}
          // Плавная анимация входа и выхода: fade + масштабирование (transition
          // 'pop'), длительность входа 220 мс, выхода 180 мс.
          transitionProps={{ transition: 'pop', duration: 220, exitDuration: 180 }}
          // Фокус после закрытия возвращаем сами — в inline-Textarea формы.
          returnFocus={false}
          onExitTransitionEnd={handleExpandedExited}
          closeButtonProps={{ 'aria-label': 'Закрыть окно ввода описания' }}
        >
          <Textarea
            label={DESCRIPTION_LABEL}
            description={`Минимум ${MIN_DESCRIPTION_LENGTH} символов`}
            placeholder={DESCRIPTION_PLACEHOLDER}
            autosize
            minRows={12}
            maxRows={20}
            value={description}
            onChange={(event) => setDescription(event.currentTarget.value)}
            error={errors.description}
            disabled={disabled}
            withAsterisk
            // FocusTrap модалки при активации сам наводит фокус на элемент
            // с data-autofocus — курсор попадает в крупный Textarea.
            data-autofocus
          />
        </Modal>
      </Stack>
    </form>
  )
}
