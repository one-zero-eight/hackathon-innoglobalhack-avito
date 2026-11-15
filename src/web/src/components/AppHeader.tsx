import { Box, Container, Group, Text } from '@mantine/core'
import { AvitoLogo } from './AvitoLogo'

/**
 * Брендинговая шапка: логотип Avito слева, подпись сервиса и радужная
 * градиентная полоса фирменной палитры внизу.
 */
export function AppHeader() {
  return (
    <Box
      component="header"
      style={{
        position: 'sticky',
        insetBlockStart: 0,
        zIndex: 100,
        backgroundColor: 'var(--mantine-color-white)',
      }}
    >
      <Container size="lg" py="sm">
        <Group justify="space-between" gap="md">
          <Group gap="md" align="center" wrap="nowrap">
            <AvitoLogo height={28} />
            <Box
              aria-hidden
              style={{
                inlineSize: 1,
                blockSize: 26,
                backgroundColor: 'var(--mantine-color-page-3)',
              }}
            />
            <Text fw={600} size="lg" lh={1.2} style={{ whiteSpace: 'nowrap' }}>
              Черновики объявлений
            </Text>
          </Group>

          <Text visibleFrom="sm" c="dimmed" size="sm">
            Ремонт и отделка · микрокатегории 101–106
          </Text>
        </Group>
      </Container>

      <Box aria-hidden className="avito-accent-line" />
    </Box>
  )
}
