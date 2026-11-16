import { Card, Group, SimpleGrid, Skeleton, Stack, Text } from '@mantine/core'
import type { Draft, Microcategory } from '../api'
import { AvitoLogo } from './AvitoLogo'
import { DraftCard } from './DraftCard'

interface ResultsListProps {
  drafts: Draft[]
  microcategories: Microcategory[]
  sourceMcId: number
  /** Сколько черновиков ещё генерируется: заглушки в той же сетке. */
  pending?: number
}

/** Заготовка на месте черновика, который ещё генерируется. */
function PendingCard() {
  return (
    <Card withBorder shadow="xs" padding="md" radius="md" bg="white">
      <Stack gap="sm">
        <Skeleton height={22} width={180} radius="md" />
        <Skeleton height={18} radius="md" />
        <Skeleton height={14} radius="md" />
        <Skeleton height={14} width="80%" radius="md" />
      </Stack>
    </Card>
  )
}

/** Список карточек черновиков (адаптивная сетка). */
export function ResultsList({
  drafts,
  microcategories,
  sourceMcId,
  pending = 0,
}: ResultsListProps) {
  return (
    <Stack gap="md">
      <Group gap="xs" justify="space-between">
        <Text fw={600} size="lg">
          {pending > 0
            ? `Подобрано черновиков: ${drafts.length} из ${drafts.length + pending}`
            : `Подобрано черновиков: ${drafts.length}`}
        </Text>
        <AvitoLogo height={16} markOnly alt="" />
      </Group>

      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md" verticalSpacing="md">
        {drafts.map((draft) => (
          <DraftCard
            key={draft.mcId}
            draft={draft}
            microcategories={microcategories}
            sourceMcId={sourceMcId}
          />
        ))}
        {Array.from({ length: pending }, (_, index) => (
          <PendingCard key={`pending-${index}`} />
        ))}
      </SimpleGrid>
    </Stack>
  )
}
