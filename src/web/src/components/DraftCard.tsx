import { Badge, Card, Group, Stack, Text } from '@mantine/core'
import { draftAccent, type Draft, type Microcategory } from '../api'

interface DraftCardProps {
  draft: Draft
  /** Список для сопоставления mcId → названия микрокатегории. */
  microcategories: Microcategory[]
  /** Исходная mcId объявления — подсветка нежелательного совпадения. */
  sourceMcId: number
}

function titleFor(microcategories: Microcategory[], mcId: number): string {
  const found = microcategories.find((item) => item.mcId === mcId)
  return found?.title ?? `Микрокатегория ${mcId}`
}

/** Карточка одного черновика: бейдж категории, заголовок и текст. */
export function DraftCard({ draft, microcategories, sourceMcId }: DraftCardProps) {
  const notGenerated = draft.title === null || draft.text === null

  return (
    <Card withBorder shadow="xs" padding="md" radius="sm" bg="white">
      <Stack gap="sm">
        <Group gap="xs">
          <Badge
            color={draftAccent(draft.mcId, sourceMcId)}
            variant="light"
            radius="sm"
            size="md"
          >
            {draft.mcId} — {titleFor(microcategories, draft.mcId)}
          </Badge>
        </Group>

        {draft.title !== null && (
          <Text fw={600} size="lg" lh={1.3} className="text-balance">
            {draft.title}
          </Text>
        )}

        {draft.text !== null && (
          <Text
            size="sm"
            c="dark.7"
            lh={1.5}
            style={{ whiteSpace: 'pre-line' }}
            className="break-words hyphens-auto"
          >
            {draft.text}
          </Text>
        )}

        {notGenerated && (
          <Badge color="gray" variant="outline" radius="sm" w="fit-content">
            Текст не сгенерирован
          </Badge>
        )}
      </Stack>
    </Card>
  )
}
