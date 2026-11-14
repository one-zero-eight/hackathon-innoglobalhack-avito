import { createTheme, type MantineColorsTuple } from '@mantine/core'

/**
 * Фирменный синий Avito (#00AAFF) и оттенки вокруг него.
 * Индекс 5 — сам брендовый цвет: на него указывает primaryShade,
 * поэтому кнопки/ссылки в filled-варианте получают ровно #00AAFF.
 */
const avitoBlue: MantineColorsTuple = [
  '#e1f7ff',
  '#c2eeff',
  '#8fe0ff',
  '#5bd2ff',
  '#2cbfff',
  '#00aaff',
  '#008fdb',
  '#0074b3',
  '#00598a',
  '#003d61',
]

/** Светло-серый фон страницы + белые поверхности карточек. */
const pageGray: MantineColorsTuple = [
  '#f8f9fa',
  '#f4f5f6',
  '#eceef0',
  '#e2e5e8',
  '#d4d8dd',
  '#c5cbd1',
  '#aeb6bf',
  '#8f99a4',
  '#6c757f',
  '#464d54',
]

const fontStack =
  "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, system-ui, sans-serif"

/**
 * Тема в стиле avito.ru: синий акцент, тёмный текст, светло-серый фон,
 * сдержанные радиусы (4–8 px) и шрифт Inter с системным fallback.
 */
export const theme = createTheme({
  primaryColor: 'avito',
  primaryShade: 5,
  defaultRadius: 'sm',
  fontFamily: fontStack,
  fontFamilyMonospace:
    "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  colors: {
    avito: avitoBlue,
    page: pageGray,
  },
  headings: {
    fontFamily: fontStack,
    fontWeight: '600',
  },
})
