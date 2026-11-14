import { Box, Card, Stack, Text, ThemeIcon } from '@mantine/core'
import { AvitoLogo } from './AvitoLogo'
import { SearchXIcon } from './icons'

/** Заглушка, когда бэкенд не предложил ни одной целевой микрокатегории. */
export function EmptyState() {
  return (
    <Card
      withBorder
      shadow="xs"
      padding="xl"
      radius="sm"
      bg="white"
      style={{ position: 'relative', overflow: 'hidden' }}
    >
      <Stack align="center" gap="xs" mih={80}>
        <ThemeIcon variant="light" color="avito" size="xl" radius="sm" aria-hidden>
          <SearchXIcon size={26} />
        </ThemeIcon>
        <Text fw={600} ta="center">
          Подходящих микрокатегорий не найдено
        </Text>
        <Text c="dimmed" size="sm" ta="center" maw={420}>
          Уточните описание услуг — и попробуйте подобрать черновики снова.
        </Text>
      </Stack>

      <Box aria-hidden className="avito-watermark">
        <AvitoLogo height={44} alt="" />
      </Box>
    </Card>
  )
}
