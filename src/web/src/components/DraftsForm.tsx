import { useCallback, useMemo, useRef, useState, type FormEvent } from 'react'
import { Alert, Button, Group, Select, Stack, Text, Textarea } from '@mantine/core'
import { fetchRandomSample, type DraftsRequest, type Microcategory } from '../api'
import type { ImportedAd } from '../adsParser'
import { AlertIcon, ShuffleIcon } from './icons'

const MIN_DESCRIPTION_LENGTH = 10

interface DraftsFormProps {
  microcategories: Microcategory[]
  /** Загрузка в процессе генерации черновиков. */
  loading: boolean
  /** Список ещё не загрузился — поля блокируем. */
  disabled?: boolean
  /** Выбранная строка загруженной таблицы. */
  selectedAd?: ImportedAd | null
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
  selectedAd = null,
  onSubmit,
}: DraftsFormProps) {
  const [description, setDescription] = useState(selectedAd?.description ?? '')
  const [mcId, setMcId] = useState<string | null>(
    selectedAd === null ? null : String(selectedAd.mcId),
  )
  const [errors, setErrors] = useState<FormErrors>(() =>
    selectedAd === null
      ? {}
      : validate(selectedAd.description, String(selectedAd.mcId), microcategories),
  )
  const [sampleLoading, setSampleLoading] = useState(false)
  const [sampleError, setSampleError] = useState<string | null>(null)

  const descriptionRef = useRef<HTMLTextAreaElement>(null)
  const mcIdRef = useRef<HTMLInputElement>(null)
  const sampleRequestId = useRef(0)

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
        <Textarea
          label="Описание объявления"
          description={`Минимум ${MIN_DESCRIPTION_LENGTH} символов`}
          placeholder="Например: выполню ремонт под ключ: штукатурка, плитка, электрика…"
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
      </Stack>
    </form>
  )
}
