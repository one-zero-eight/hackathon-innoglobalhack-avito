import { Group, SimpleGrid, Stack, Text } from '@mantine/core'
import type { Draft, Microcategory } from '../api'
import { AvitoLogo } from './AvitoLogo'
import { DraftCard } from './DraftCard'

interface ResultsListProps {
  drafts: Draft[]
  microcategories: Microcategory[]
  sourceMcId: number
}

/** Список карточек черновиков (адаптивная сетка). */
export function ResultsList({ drafts, microcategories, sourceMcId }: ResultsListProps) {
  return (
    <Stack gap="md">
      <Group gap="xs" justify="space-between">
        <Text fw={600} size="lg">
          Подобрано черновиков: {drafts.length}
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
      </SimpleGrid>
    </Stack>
  )
}
