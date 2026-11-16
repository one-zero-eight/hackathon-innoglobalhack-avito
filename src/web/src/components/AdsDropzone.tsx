import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { Alert, Box, Button, Group, Select, Stack, Text } from '@mantine/core'
import { parseAdsFile, type ImportedAd } from '../adsParser'
import { AlertIcon } from './icons'

const MAX_FILE_SIZE = 10 * 1024 * 1024
const DEFAULT_CONCURRENCY = 4
const CONCURRENCY_OPTIONS = Array.from({ length: 8 }, (_, index) => ({
  value: String(index + 1),
  label: String(index + 1),
}))

interface AdsDropzoneProps {
  /** Запуск пакетной обработки импортированного массива объявлений. */
  onProcessAll: (ads: ImportedAd[], concurrency: number) => void
  /** Пакетная обработка уже идёт — кнопка запуска блокируется. */
  bulkRunning: boolean
}

function errorMessage(error: unknown): string {
  return error instanceof Error && error.message
    ? error.message
    : 'Не удалось прочитать файл. Проверьте его формат и попробуйте ещё раз.'
}

/** Загрузка объявлений из таблицы и запуск пакетной обработки. */
export function AdsDropzone({ onProcessAll, bulkRunning }: AdsDropzoneProps) {
  const [ads, setAds] = useState<ImportedAd[]>([])
  const [fileName, setFileName] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [concurrency, setConcurrency] = useState<string | null>(String(DEFAULT_CONCURRENCY))
  const inputRef = useRef<HTMLInputElement>(null)
  const dragDepth = useRef(0)
  const requestId = useRef(0)

  const handleFile = useCallback(async (file: File) => {
    const currentRequest = ++requestId.current
    setLoading(true)
    setError(null)
    setAds([])
    setFileName(null)

    try {
      if (file.size > MAX_FILE_SIZE) {
        throw new Error('Файл слишком большой. Максимальный размер — 10 МБ.')
      }
      const parsed = parseAdsFile(await file.text(), file.name)
      if (currentRequest === requestId.current) {
        setAds(parsed)
        setFileName(file.name)
      }
    } catch (fileError: unknown) {
      if (currentRequest === requestId.current) {
        setError(errorMessage(fileError))
      }
    } finally {
      if (currentRequest === requestId.current) {
        setLoading(false)
      }
    }
  }, [])

  const handleInput = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0]
    if (file) {
      void handleFile(file)
    }
    event.currentTarget.value = ''
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    const file = event.dataTransfer.files[0]
    if (file) {
      void handleFile(file)
    }
  }

  return (
    <Stack gap="md">
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.json,.jsonl,text/csv,application/json"
        onChange={handleInput}
        hidden
      />

      <Box
        p="lg"
        bg={dragging ? 'avito.0' : 'gray.0'}
        aria-busy={loading}
        onDragEnter={(event) => {
          event.preventDefault()
          dragDepth.current += 1
          setDragging(true)
        }}
        onDragOver={(event) => {
          event.preventDefault()
          event.dataTransfer.dropEffect = 'copy'
        }}
        onDragLeave={() => {
          dragDepth.current = Math.max(0, dragDepth.current - 1)
          if (dragDepth.current === 0) {
            setDragging(false)
          }
        }}
        onDrop={handleDrop}
        style={{
          border: `2px dashed var(--mantine-color-${dragging ? 'avito-5' : 'gray-4'})`,
          borderRadius: 'var(--mantine-radius-md)',
          textAlign: 'center',
        }}
      >
        <Stack align="center" gap="sm">
          <Text fw={600}>Перетащите таблицу объявлений сюда</Text>
          <Text size="sm" c="dimmed" maw={440}>
            CSV с разделителем «;» или «,», JSON-массив либо JSONL. Обязательные поля:
            description и mcId.
          </Text>
          <Button
            type="button"
            variant="light"
            loading={loading}
            onClick={() => inputRef.current?.click()}
          >
            Выбрать файл
          </Button>
        </Stack>
      </Box>

      {error !== null && (
        <Alert
          color="red"
          variant="light"
          radius="md"
          icon={<AlertIcon size={18} />}
          title="Не удалось загрузить объявления"
          withCloseButton
          onClose={() => setError(null)}
        >
          <Text size="sm">{error}</Text>
        </Alert>
      )}

      {ads.length > 0 && fileName !== null && (
        <Stack gap="sm">
          <Group justify="space-between" align="baseline">
            <Text size="sm" fw={600}>
              Загружено объявлений: {ads.length}
            </Text>
            <Text size="xs" c="dimmed" className="break-words">
              {fileName}
            </Text>
          </Group>

          <Group justify="space-between" align="flex-end" gap="md">
            <Button
              type="button"
              loading={bulkRunning}
              disabled={bulkRunning}
              onClick={() =>
                onProcessAll(
                  ads.map((ad) => ({ ...ad })),
                  Number(concurrency ?? DEFAULT_CONCURRENCY),
                )
              }
            >
              Обработать все
            </Button>
            <Select
              label="Параллелизм"
              description="Сколько объявлений обрабатывать одновременно (1–8)"
              data={CONCURRENCY_OPTIONS}
              value={concurrency}
              onChange={setConcurrency}
              allowDeselect={false}
              disabled={bulkRunning}
              maw={240}
              size="sm"
            />
          </Group>
        </Stack>
      )}
    </Stack>
  )
}
