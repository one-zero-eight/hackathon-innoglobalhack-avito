import { useCallback, useMemo, useState, type FormEvent } from 'react'
import { Button, Group, Select, Stack, Textarea } from '@mantine/core'
import type { DraftsRequest, Microcategory } from '../api'

const MIN_DESCRIPTION_LENGTH = 10

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

function validate(description: string, mcId: string | null): FormErrors {
  const errors: FormErrors = {}

  const trimmed = description.trim()
  if (trimmed.length === 0) {
    errors.description = 'Укажите описание объявления.'
  } else if (trimmed.length < MIN_DESCRIPTION_LENGTH) {
    errors.description = `Описание слишком короткое: минимум ${MIN_DESCRIPTION_LENGTH} символов.`
  }

  if (mcId === null || mcId === '') {
    errors.mcId = 'Выберите текущую микрокатегорию.'
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
      const nextErrors = validate(description, mcId)
      setErrors(nextErrors)
      if (nextErrors.description || nextErrors.mcId) {
        return
      }
      onSubmit({ description: description.trim(), mcId: Number(mcId) })
    },
    [description, mcId, onSubmit],
  )

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
        />

        <Group>
          <Button type="submit" loading={loading} disabled={disabled}>
            Подобрать черновики
          </Button>
        </Group>
      </Stack>
    </form>
  )
}
