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

/**
 * Акценты фирменной палитры Avito — цвета четырёх точек официального
 * логотипа (публичный SVG из шапки avito.ru): зелёный #04E061,
 * коралловый #FF4053, фиолетовый #965EEB. Индекс 5 — сам брендовый цвет,
 * индексы 0/8 — пара «фон бейджа / текст бейджа» с контрастом не ниже
 * WCAG AA (проверено: 6.2:1, 9.1:1, 9.5:1 на белом фоне).
 */
const brandGreen: MantineColorsTuple = [
  '#e6fdee',
  '#cbfbe0',
  '#99f4c3',
  '#5cedab',
  '#2ae48f',
  '#04e061',
  '#03c053',
  '#029641',
  '#026b2f',
  '#01421d',
]

const brandCoral: MantineColorsTuple = [
  '#fff0f1',
  '#ffdfe2',
  '#ffc0c7',
  '#ff9aa6',
  '#ff6c7c',
  '#ff4053',
  '#e01e32',
  '#b31526',
  '#850f1c',
  '#590a13',
]

const brandViolet: MantineColorsTuple = [
  '#f3edfd',
  '#e7dbfb',
  '#d0bcf7',
  '#b798f2',
  '#a17bee',
  '#965eeb',
  '#7f46d4',
  '#6733ab',
  '#4f2683',
  '#371a5c',
]

/**
 * Жёлтый акцент — четвёртый яркий цвет брендовых материалов Avito
 * (используется в продуктах сервиса). Пара 0/8 даёт 5.5:1 — AA для
 * обычного текста.
 */
const brandYellow: MantineColorsTuple = [
  '#fffae6',
  '#fff3c2',
  '#ffe98a',
  '#ffdd4d',
  '#ffd91f',
  '#ffd400',
  '#e0b800',
  '#b39300',
  '#7a6400',
  '#4d3f00',
]

/** Глубокий синий Avito (7-й оттенок основного бренда) — для «Сантехники». */
const brandSea: MantineColorsTuple = [
  '#e1f1fa',
  '#c4e2f3',
  '#93c9e8',
  '#5dadd8',
  '#2b93c9',
  '#0074b3',
  '#00639a',
  '#005180',
  '#003d61',
  '#002b45',
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

/** Путь к официальным файлам логотипа — чтобы не хардкодить их в компонентах. */
export const brandAssets = {
  /** Полный логотип: точки + вордмарк «Avito». */
  logo: '/avito-logo.svg',
  /** Компактный знак: только четыре цветные точки. */
  mark: '/favicon.svg',
} as const

/** Название палитры-акцента из `theme.colors`. */
export type BrandPalette =
  | 'avito'
  | 'brandSea'
  | 'brandGreen'
  | 'brandCoral'
  | 'brandViolet'
  | 'brandYellow'

/**
 * Фирменная цветовая кодировка микрокатегорий раздела «Ремонт и отделка»:
 * каждой микрокатегории — своя пара цветов из палитры Avito.
 */
const microcategoryPalettes: Record<number, BrandPalette> = {
  101: 'avito',
  102: 'brandViolet',
  103: 'brandGreen',
  104: 'brandSea',
  105: 'brandYellow',
  106: 'brandCoral',
}

/** Палитра-акцент для микрокатегории; неизвестные id — фирменный синий. */
export function microcategoryPalette(mcId: number): BrandPalette {
  return microcategoryPalettes[mcId] ?? 'avito'
}

/**
 * Готовая пара цветов для бейджа микрокатегории: светлый фон (оттенок 0)
 * и тёмный текст (оттенок 8) — контраст не ниже WCAG AA.
 */
export function badgeColors(palette: BrandPalette): { bg: string; text: string } {
  return {
    bg: `var(--mantine-color-${palette}-0)`,
    text: `var(--mantine-color-${palette}-8)`,
  }
}

/** Цвет цветной полосы-акцента (оттенок 5 — чистый брендовый цвет). */
export function accentColor(palette: BrandPalette): string {
  return `var(--mantine-color-${palette}-5)`
}

/**
 * Тема в стиле avito.ru: синий акцент, тёмный текст, светло-серый фон,
 * сдержанные радиусы (4–8 px) и шрифт Inter с системным fallback.
 * Все фирменные цвета объявлены здесь палитрами — компоненты обращаются
 * к ним через CSS-переменные Mantine, hex в компонентах не появляется.
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
    brandGreen,
    brandCoral,
    brandViolet,
    brandYellow,
    brandSea,
  },
  headings: {
    fontFamily: fontStack,
    fontWeight: '700',
  },
  components: {
    /*
     * Тач-цели: высота кнопок не меньше 44 px (по умолчанию size="sm" —
     * 36 px). Высоту полей ввода перекрыть здесь нельзя: она считается
     * на InputWrapper, поэтому правило живёт в index.css.
     */
    Button: {
      vars: () => ({
        root: {
          '--button-height-sm': '2.75rem',
          '--button-height-md': '2.75rem',
        },
      }),
    },
  },
})
