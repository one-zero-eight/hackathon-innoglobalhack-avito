import { Badge, Box, Card, Group, Stack, Text } from '@mantine/core'
import { accentColor, badgeColors, microcategoryPalette } from '../theme'
import type { Draft, Microcategory } from '../api'

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

/**
 * Карточка одного черновика: цветная топ-полоска и бейдж микрокатегории
 * в фирменной паре цветов, заголовок и текст объявления.
 */
export function DraftCard({ draft, microcategories, sourceMcId }: DraftCardProps) {
  const notGenerated = draft.title === null || draft.text === null
  const sameAsSource = draft.mcId === sourceMcId

  // Совпадение с исходной категорией — ошибка подбора, подсвечиваем красным.
  const palette = sameAsSource ? null : microcategoryPalette(draft.mcId)
  const badge = palette === null ? badgeColors('brandCoral') : badgeColors(palette)
  const strip = palette === null ? accentColor('brandCoral') : accentColor(palette)

  return (
    <Card
      withBorder
      shadow="xs"
      padding={0}
      radius="sm"
      bg="white"
      className="avito-card"
      style={{ overflow: 'hidden' }}
    >
      <Box aria-hidden className="avito-card-accent" style={{ background: strip }} />

      <Stack gap="sm" p="md">
        <Group gap="xs">
          <Badge
            radius="sm"
            size="md"
            styles={{
              root: { backgroundColor: badge.bg, color: badge.text },
            }}
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

        {sameAsSource && (
          <Text size="xs" c="brandCoral.8">
            Совпало с исходной микрокатегорией — проверьте подбор.
          </Text>
        )}
      </Stack>
    </Card>
  )
}
