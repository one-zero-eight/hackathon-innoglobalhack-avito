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
        <Group justify="space-between" align="center" gap="lg">
          <Stack gap={4}>
            <AvitoLogo height={22} />
          </Stack>

          <Text size="xs" c="dimmed">
            © {CURRENT_YEAR} 108 двоеточие 3
          </Text>
        </Group>
      </Container>
    </Box>
  )
}
