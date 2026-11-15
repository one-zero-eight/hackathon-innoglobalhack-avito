import { useState } from 'react'
import {
  Alert,
  Badge,
  Box,
  Button,
  Group,
  Progress,
  ScrollArea,
  SimpleGrid,
  Stack,
  Table,
  Text,
} from '@mantine/core'
import type { BatchDoneEvent, Microcategory } from '../api'
import type { ImportedAd } from '../adsParser'
import { downloadBulkCsv, downloadBulkJson } from '../bulkExport'
import type { BulkPhase, BulkRow, BulkRowStatus } from '../bulkTypes'
import { DraftCard } from './DraftCard'
import { AlertIcon } from './icons'

interface BulkResultsProps {
  /** Строки входного массива (в порядке файла). */
  ads: ImportedAd[]
  rows: BulkRow[]
  phase: BulkPhase
  /** Ожидаемое число строк (из кадра start; fallback — длина массива). */
  total: number
  /** Итоговый кадр done, если уже пришёл. */
  summary: BatchDoneEvent | null
  /** Глобальная ошибка потока (сеть, протокол) при phase=failed. */
  globalError: string | null
  microcategories: Microcategory[]
  onCancel: () => void
}

interface StatusBadge {
  color: string
  label: string
}

/** Подпись и цвет бейджа статуса строки с учётом остановки задачи. */
function statusBadge(status: BulkRowStatus, phase: BulkPhase): StatusBadge {
  switch (status) {
    case 'pending':
      return phase === 'cancelled'
        ? { color: 'gray', label: 'Остановлено' }
        : { color: 'gray', label: 'Ожидание' }
    case 'in_progress':
      return { color: 'blue', label: 'В работе' }
    case 'done':
      return { color: 'green', label: 'Готово' }
    case 'error':
      return { color: 'red', label: 'Ошибка' }
  }
}

/** Пакетные результаты: прогресс, таблица по строкам файла, экспорт. */
export function BulkResults({
  ads,
  rows,
  phase,
  total,
  summary,
  globalError,
  microcategories,
  onCancel,
}: BulkResultsProps) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  const processed = rows.filter((row) => row.status === 'done' || row.status === 'error').length
  const succeeded = rows.filter((row) => row.status === 'done').length
  const failed = rows.filter((row) => row.status === 'error').length
  const denominator = Math.max(total, rows.length)
  const percent = denominator > 0 ? (processed / denominator) * 100 : 0

  const toggleRow = (index: number) => {
    setExpanded((current) => {
      const next = new Set(current)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  const header = (
    <Group justify="space-between" align="flex-end" gap="sm">
      <Stack gap={2}>
        <Text fw={600} size="lg">
          Пакетная обработка: {ads.length} объявлений
        </Text>
        {phase === 'done' && summary !== null && (
          <Text size="sm" c="dimmed">
            Итог: обработано {summary.processed}, успешно {summary.succeeded}, ошибок{' '}
            {summary.failed}.
          </Text>
        )}
        {phase === 'cancelled' && (
          <Text size="sm" c="dimmed">
            Остановлено по вашей команде: обработано {processed} из {denominator}.
          </Text>
        )}
      </Stack>

      <Group gap="xs">
        {phase === 'running' && (
          <Button type="button" variant="light" color="gray" size="sm" onClick={onCancel}>
            Остановить
          </Button>
        )}
        <Button
          type="button"
          variant="light"
          size="sm"
          disabled={processed === 0}
          onClick={() => downloadBulkJson(ads, rows)}
        >
          Скачать JSON
        </Button>
        <Button
          type="button"
          variant="light"
          size="sm"
          disabled={processed === 0}
          onClick={() => downloadBulkCsv(ads, rows)}
        >
          Скачать CSV
        </Button>
      </Group>
    </Group>
  )

  const progress = (
    <Stack gap={6}>
      <Progress
        value={percent}
        animated={phase === 'running'}
        size="lg"
        radius="xl"
        aria-label={`Обработано ${processed} из ${denominator}`}
      />
      <Group justify="space-between" gap="xs">
        <Text size="sm" c="dimmed">
          Обработано {processed} из {denominator}
        </Text>
        <Group gap="xs">
          <Badge color="green" variant="light" radius="md">
            Готово: {succeeded}
          </Badge>
          <Badge color="red" variant="light" radius="md">
            Ошибок: {failed}
          </Badge>
        </Group>
      </Group>
    </Stack>
  )

  const table = (
    <ScrollArea.Autosize
      mah={560}
      type="auto"
      viewportProps={{
        tabIndex: 0,
        role: 'region',
        'aria-label': 'Таблица результатов пакетной обработки',
      }}
    >
      <Table striped highlightOnHover stickyHeader miw={640} withTableBorder captionSide="top">
        <Table.Caption>Результаты по строкам файла</Table.Caption>
        <Table.Thead>
          <Table.Tr>
            <Table.Th scope="col">№</Table.Th>
            <Table.Th scope="col">Описание</Table.Th>
            <Table.Th scope="col">Исходная mcId / категория</Table.Th>
            <Table.Th scope="col">Статус</Table.Th>
            <Table.Th scope="col">Черновики</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {ads.map((ad, index) => {
            const row = rows[index] ?? { status: 'pending', drafts: [], error: null }
            const badge = statusBadge(row.status, phase)
            const category = microcategories.find((item) => item.mcId === ad.mcId)
            const details = row.drafts.length > 0 || row.error !== null

            return [
              <Table.Tr key={`row-${index}`}>
                <Table.Td>
                  <Text size="sm">{index + 1}</Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm" lineClamp={2} maw={360}>
                    {ad.description}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">
                    {ad.mcId}
                    {category ? ` — ${category.title}` : ''}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Badge color={badge.color} variant="light" radius="md">
                    {badge.label}
                  </Badge>
                  {row.error !== null && (
                    <Text size="xs" c="red.8" mt={4} maw={260} className="break-words">
                      {row.error}
                    </Text>
                  )}
                </Table.Td>
                <Table.Td>
                  {details ? (
                    <Button
                      type="button"
                      size="sm"
                      variant="subtle"
                      aria-expanded={expanded.has(index)}
                      onClick={() => toggleRow(index)}
                    >
                      {expanded.has(index)
                        ? 'Скрыть'
                        : row.drafts.length > 0
                          ? `Показать (${row.drafts.length})`
                          : 'Подробнее'}
                    </Button>
                  ) : (
                    <Text size="sm" c="dimmed">
                      —
                    </Text>
                  )}
                </Table.Td>
              </Table.Tr>,

              expanded.has(index) && (
                <Table.Tr key={`details-${index}`}>
                  <Table.Td colSpan={5}>
                    <Box py={4}>
                      {row.drafts.length > 0 && (
                        <SimpleGrid
                          cols={{ base: 1, sm: 2 }}
                          spacing="md"
                          verticalSpacing="md"
                        >
                          {row.drafts.map((draft) => (
                            <DraftCard
                              key={draft.mcId}
                              draft={draft}
                              microcategories={microcategories}
                              sourceMcId={ad.mcId}
                            />
                          ))}
                        </SimpleGrid>
                      )}
                      {row.error !== null && row.drafts.length === 0 && (
                        <Text size="sm" c="red.8">
                          {row.error}
                        </Text>
                      )}
                      {row.error === null && row.drafts.length === 0 && (
                        <Text size="sm" c="dimmed">
                          Черновики ещё не получены.
                        </Text>
                      )}
                    </Box>
                  </Table.Td>
                </Table.Tr>
              ),
            ]
          })}
        </Table.Tbody>
      </Table>
    </ScrollArea.Autosize>
  )

  return (
    <Stack gap="md">
      {header}
      {progress}

      {phase === 'failed' && globalError !== null && (
        <Alert color="red" title="Пакетная обработка прервана" radius="md" icon={<AlertIcon />}>
          <Text size="sm">{globalError}</Text>
        </Alert>
      )}

      {table}
    </Stack>
  )
}
