import { Card, Stack, Text, ThemeIcon } from '@mantine/core'

/** Заглушка, когда бэкенд не предложил ни одной целевой микрокатегории. */
export function EmptyState() {
  return (
    <Card withBorder shadow="xs" padding="xl" radius="sm" bg="white">
      <Stack align="center" gap="xs" mih={80}>
        <ThemeIcon variant="light" color="page" size="lg" radius="sm" aria-hidden>
          ?
        </ThemeIcon>
        <Text fw={600} ta="center">
          Подходящих микрокатегорий не найдено
        </Text>
        <Text c="dimmed" size="sm" ta="center" maw={420}>
          Уточните описание услуг — и попробуйте подобрать черновики снова.
        </Text>
      </Stack>
    </Card>
  )
}
