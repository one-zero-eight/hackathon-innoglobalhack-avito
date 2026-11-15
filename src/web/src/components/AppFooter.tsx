import { Box, Container, Group, Stack, Text } from '@mantine/core'
import { AvitoLogo } from './AvitoLogo'

const CURRENT_YEAR = new Date().getFullYear()

/** Футер страницы: логотип, дисклеймер и копирайт. */
export function AppFooter() {
  return (
    <Box
      component="footer"
      mt="xl"
      style={{
        borderTop: '1px solid var(--mantine-color-page-3)',
        backgroundColor: 'var(--mantine-color-white)',
      }}
    >
      <Container size="lg" py="md">
        <Group justify="space-between" align="flex-start" gap="lg">
          <Stack gap={4}>
            <AvitoLogo height={22} />
            <Text size="xs" c="dimmed">
              Учебный проект хакатона InnoGlobalHack × Avito. Логотип и фирменные
              цвета — товарные знаки Avito, используются только в демонстрационных
              целях.
            </Text>
          </Stack>

          <Text size="xs" c="dimmed">
            © {CURRENT_YEAR} Команда InnoGlobalHack
          </Text>
        </Group>
      </Container>
    </Box>
  )
}
